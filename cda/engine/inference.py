import os
import torch
import torch.utils.data
import logging
from tqdm import tqdm
import numpy as np
from cda.data.build import make_data_loader
from cda.data.datasets.evaluation import evaluate, evaluate_adv
from cda.utils import dist_util, mkdir
dir_path = os.path.dirname(os.path.realpath(__file__))


def normalize_fn(t):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):

    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("CDA.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, eval_dataset, num_images, device, cfg,
                       save_feats, train_config, eval_model, save_layer_index):

    idx = 0
    predictions, gt_labels = [], []

    for batch in tqdm((data_loader)):
        images, targets, image_ids = batch

        assert torch.max(images) <= 1, False
        assert torch.min(images) >= 0, False

        with torch.no_grad():

            normalized_img = normalize_fn(images.to(device).detach())
            outputs = model(normalized_img)
            pred = outputs.argmax(dim=-1)
            predictions.append(pred.cpu().numpy())
            gt_labels.append(targets)

        idx += images.shape[0]

        if idx >= num_images:
            break

    predictions = np.concatenate(predictions)
    gt_labels = np.concatenate(gt_labels)

    return np.array(predictions), gt_labels


def inference(model, data_loader, eval_dataset, output_folder, num_images=-1, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("CDA.inference")
    dataset = data_loader.dataset
    num_images = num_images if num_images != -1 else len(dataset)
    logger.info("Evaluating Normal {} dataset({} images):".format(eval_dataset, num_images))
    predictions, gt_labels = compute_on_dataset(model, data_loader, eval_dataset, num_images, **kwargs)

    return evaluate(dataset=dataset, gt_labels=gt_labels, predictions_clean=predictions,
                    output_dir=output_folder), predictions


# @torch.no_grad()
def do_evaluation(cfg, model, distributed, **kwargs):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()

    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)

    eval_results = []
    predictions = []

    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result, predictions_ = inference(model, data_loader, dataset_name, output_folder, cfg=cfg,
                                              device=device, **kwargs)
        eval_results.append(eval_result)
        predictions.append(predictions_)
    return eval_results, predictions


def compute_on_dataset_adv(model, data_loader, eval_dataset, num_images, perturbmode, generator,
                           eps, device, cfg, save_feats, train_config, eval_model, save_layer_index):

    idx = 0
    predictions_adv, gt_labels, predictions_clean = [], [], []

    for batch in tqdm(data_loader):

        images, targets, image_ids = batch
        images = images.cuda()
        images_org = images.clone()

        assert torch.max(images) <= 1, 'Wrong Normalization'
        assert torch.min(images) >= 0, 'Wrong Normalization'

        image_adv1 = generator(images)

        if image_adv1.shape[-1] != images.shape[-1]:
            image_adv1 = torch.nn.functional.interpolate(image_adv1, size=images.shape[2:], mode='bilinear')

        adv1 = torch.min(torch.max(image_adv1, images - eps), images + eps)
        images_adv = torch.clamp(adv1, 0.0, 1.0)

        with torch.no_grad():
            normalized_img = normalize_fn(images_adv.to(device).clone())
            outputs_adv = model(normalized_img)
            outputs_clean = model(normalize_fn(images_org.to(device).clone()))

        pred_adv = outputs_adv.argmax(dim=-1)
        predictions_adv.append(pred_adv.cpu().numpy())
        gt_labels.append(targets)

        pred_clean = outputs_clean.argmax(dim=-1)
        predictions_clean.append(pred_clean.cpu().numpy())
        idx += images.shape[0]

        if idx >= num_images:
            break

    predictions_adv = np.concatenate(predictions_adv)
    gt_labels = np.concatenate(gt_labels)
    predictions_clean = np.concatenate(predictions_clean)

    return predictions_adv, gt_labels, predictions_clean


def inference_adv(model, data_loader, eval_dataset, output_folder, num_images=-1, perturbmode=False, **kwargs):

    dataset = data_loader.dataset
    num_images = num_images if num_images != -1 else len(dataset)

    logger = logging.getLogger("CDA.inference")
    logger.info("Evaluating Adversarial images of {} dataset({} images) with perturb mode:{} :".format(
        eval_dataset, num_images, perturbmode))

    predictions, gt_labels, predictions_clean = compute_on_dataset_adv(
        model, data_loader, eval_dataset, num_images, perturbmode, **kwargs)

    return evaluate_adv(dataset=dataset, gt_labels=gt_labels, predictions_adv=predictions,
                        predictions_clean=predictions_clean, output_dir=output_folder), predictions


@ torch.no_grad()
def do_evaluation_adv(cfg, model, distributed, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(
        cfg, is_train=False, distributed=distributed, shuffle=False)
    eval_results = []
    predictions = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_adv", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result, predictions_ = inference_adv(
            # model, data_loader, dataset_name, device, generator, eps, mean, perturbmode, output_folder,
            # save_feats, cfg, det_name, pooling_type, num_images, ** kwargs)
            model, data_loader, dataset_name, output_folder, cfg=cfg, device=device, **kwargs)

        eval_results.append(eval_result)

        predictions.append(predictions_)
    return eval_results, predictions
