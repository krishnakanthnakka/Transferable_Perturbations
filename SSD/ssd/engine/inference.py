import logging
import os
import torch
import torch.utils.data
import torch

from tqdm import tqdm
from torchvision.utils import save_image
from ssd.data.build import make_data_loader
from ssd.data.datasets.evaluation import evaluate
from ssd.utils import dist_util, mkdir
from ssd.utils.dist_util import synchronize, is_main_process
from ssd.data.datasets import VOCDataset
from .inference_utils import subtract_mean, add_mean, normalize_to_minus_1_to_plus_1
from icecream import ic

class_names = VOCDataset.class_names


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, device, cfg, save_feats, dataset_name,
                       mean, det_name, pooling_type, num_images):

    results_dict = {}
    idx = 0

    logger = logging.getLogger("SSD.inference")
    logger.info("Model is set with is_training flag: {}".format(model.training))

    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images.to(device))
            outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update({int(img_id): result for img_id, result in zip(image_ids, outputs)})
        idx += images.shape[0]
        if idx > num_images:
            break

    return results_dict


def inference(model, data_loader, dataset_name, device, output_folder=None, cfg=None, save_feats=False,
              mean=None, det_name=None, pooling_type=None, num_images=-1, use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    num_images = num_images if num_images != -1 else len(dataset)

    logger.info("Evaluating Normal images of {} dataset({} images):".format(
        dataset_name, num_images))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(
            model, data_loader, device, cfg, save_feats, dataset_name, mean, det_name, pooling_type, num_images)
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder,
                    num_images=num_images, **kwargs)


@ torch.no_grad()
def do_evaluation(cfg, model, distributed, save_feats, mean, det_name, pooling_type, num_images, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(
        cfg, is_train=False, distributed=distributed)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference(
            model, data_loader, dataset_name, device, output_folder, cfg, save_feats, mean, det_name,
            pooling_type, num_images, **kwargs)
        eval_results.append(eval_result)
    return eval_results


def compute_on_dataset_adv(model, data_loader, device, generator, eps, mean, save_feats, cfg,
                           dataset_name, det_name, pooling_type, num_images, perturbmode, gen_size):

    results_dict = {}
    idx = 0
    logger = logging.getLogger("SSD.inference")

    for batch in tqdm(data_loader):

        images, targets, image_ids = batch
        images = images.cuda()
        images_255 = add_mean(images, mean)
        images_1 = normalize_to_minus_1_to_plus_1(images_255)
        image512_clean1 = torch.nn.functional.interpolate(images_1, size=(512, 512), mode='bilinear')

        gen_shape = (gen_size, gen_size)  # {(300, 300),(224, 224)}
        img_shape = images_255.shape[2:]
        image512_clean1 = torch.nn.functional.interpolate(images_255 / 255.0, size=gen_shape, mode='bilinear')
        image512_adv1 = generator(image512_clean1)

        adv1 = torch.nn.functional.interpolate(image512_adv1, size=img_shape, mode='bilinear')
        adv1 = torch.min(torch.max(adv1, images_255 / 255.0 - eps), images_255 / 255 + eps)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
        adv255 = adv1 * 255
        images_adv = subtract_mean(adv255, mean)

        if 1 and idx == 0:
            save_image(adv1, 'demo.png')
            logger.info("Epsilon: {}, generator input resolution: {}".format(eps * 255.0, gen_shape))

        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images_adv.to(device))
            outputs = [o.to(cpu_device) for o in outputs]
            outputs_clean = model(images.to(device))
            outputs_clean = [o.to(cpu_device) for o in outputs_clean]

        results_dict.update({int(img_id): result for img_id, result in zip(image_ids, outputs)})

        idx += images.shape[0]
        if idx > num_images:
            break

    return results_dict


def inference_adv(model, data_loader, dataset_name, device, generator, eps, mean, perturbmode,
                  output_folder=None, save_feats=False, cfg=None, det_name=None, pooling_type=None,
                  num_images=-1, use_cached=False, gen_size=224, **kwargs):

    dataset = data_loader.dataset

    num_images = num_images if num_images != -1 else len(dataset)

    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating Adversarial images of {} dataset({} images) with perturb mode:{} :".format(
        dataset_name, num_images, perturbmode))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset_adv(
            model, data_loader, device, generator, eps, mean, save_feats, cfg, dataset_name, det_name,
            pooling_type, num_images, perturbmode, gen_size)
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder,
                    num_images=num_images, **kwargs)


@ torch.no_grad()
def do_evaluation_adv(cfg, model, distributed, generator, eps, mean, save_feats, det_name,
                      pooling_type, num_images, perturbmode, gen_size, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    generator.eval()

    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False,
                                        distributed=distributed)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_adv",
                                     dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference_adv(model, data_loader, dataset_name, device, generator,
                                    eps, mean, perturbmode, output_folder, save_feats,
                                    cfg, det_name, pooling_type, num_images, gen_size=gen_size, ** kwargs)
        eval_results.append(eval_result)
    return eval_results
