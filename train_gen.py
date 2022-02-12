import time
import torch
import random
import numpy as np

import PIL
import torchvision
import os
import sys
import datetime
import pprint
import logging
import warnings

from options.train_options1 import TrainOptions
from torch.utils.data import DataLoader
from models import create_model
from cda.utils.utils import AverageMeter, create_logger
from pprint import pformat
from cda.config import cfg as detcfg
from cda.data.build import make_data_loader
from cda.utils.logger import setup_logger
from cda.utils import dist_util, mkdir, savefiles
from cda.utils.metric_logger import MetricLogger

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
basedir = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':

    command_line = 'python ' + ' '.join(sys.argv)
    opt = TrainOptions().parse()

    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    detcfg.merge_from_file('./cda/config/{}_{}.yaml'.format(opt.train_classifier, opt.train_dataset))

    opt.softmax2D = False
    opt.data_dim = 'high'
    opt.eps = 10.0 / 255.0  # CHANGED
    opt.lr = 0.0002  # CHANGED

    if not opt.pretrained_netG:
        opt.pretrain_weights = ''

    opt.train_classifier_weights = ''
    detcfg.MODEL.BACKBONE.PRETRAINED = True
    detcfg.MODEL.NUM_CLASSES = 1000

    opt.classifier_weights = ''
    opt.beta1 = 0.5
    opt.lr_gamma = 0.3

    eval_freq = opt.save_epoch_freq
    opt.print_freq = 100
    opt.save_latest_freq = 5000
    opt.save_by_iter = True

    # IMAGE SIZE
    if detcfg.MODEL.META_ARCHITECTURE == 'inception_v3':
        detcfg.INPUT.RESIZE_SIZE = 300
        detcfg.INPUT.IMAGE_SIZE = 299
        opt.train_getG_299 = True
    else:
        detcfg.INPUT.RESIZE_SIZE = 256
        detcfg.INPUT.IMAGE_SIZE = 224
        opt.train_getG_299 = False

    detcfg.SOLVER.BATCH_SIZE = 16
    detcfg.TEST.CONFIDENCE_THRESHOLD = 0.5

    detcfg.OUTPUT_DIR = os.path.join(basedir, "checkpoints", "{}_{}_{}".format(
        opt.train_dataset, opt.train_classifier, opt.loss_type))

    detcfg.freeze()

    if detcfg.OUTPUT_DIR:
        mkdir(detcfg.OUTPUT_DIR)

    opt.detcfg = detcfg
    opt.detckpt = ''
    opt.reqd_class_index = 0
    opt.weightfile = ''
    opt.continue_train = False
    opt.load_iter = 6000
    opt.attackobjective = 'Blind'
    opt.input_nc = 3

    opt.weight_L2 = 0
    opt.weight_ce = 1
    opt.weight_rl = 1
    opt.weight_att = 1
    opt.weight_feat = 1
    opt.perturbmode = False
    opt.stop_iter = 550000000
    opt.pooling_type = 'Full'
    opt.num_images = 5000

    logger = setup_logger("CDA", dist_util.get_rank(), opt.detcfg.OUTPUT_DIR)
    logger.info("Command line: {}".format(command_line))

    logger.info("Experiment started at {}".format(datetime.datetime.now()
                                                          .strftime("%H:%M:%S secs on %d/%m/%y")))

    logger.info("Environment:")
    logger.info("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.info("\tPyTorch: {}".format(torch.__version__))
    logger.info("\tTorchvision: {}".format(torchvision.__version__))
    logger.info("\tCUDA: {}".format(torch.version.cuda))
    logger.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.info("\tNumPy: {}".format(np.__version__))
    logger.info("\tPIL: {}".format(PIL.__version__))
    logger.info(pformat(vars(opt)))

    model = create_model(opt)
    model.setup(opt)

    dataloader = make_data_loader(opt.detcfg, is_train=True, distributed=False,
                                  max_iter=1e+10, start_iter=0, shuffle=opt.data_shuffle,
                                  data_aug=opt.data_aug)

    opt.dataset = dataloader.dataset

    total_iters = 0
    avg_loss_ce = AverageMeter()
    avg_loss_rl = AverageMeter()
    avg_loss_att = AverageMeter()
    avg_loss_L2 = AverageMeter()
    avg_loss_feat = AverageMeter()

    niter = 0

    logger = logging.getLogger("CDA.trainer")
    logger.info("Start training ...")
    savefiles(detcfg.OUTPUT_DIR, opt)

    num_iterations_per_epoch = int(len(dataloader.dataset) / detcfg.SOLVER.BATCH_SIZE)

    logger.info("Number of iterations per epoch: {}, Total Images:{}".format(num_iterations_per_epoch,
                                                                             len(dataloader.dataset)))
    logger.info("epsilon: {:.1f}, seed:{}, warm start eps: {}, warm start L2 steps: {}".format(
        opt.eps * 255, seed, opt.warm_start, opt.warm_start_L2_steps))

    epoch_iter = 0
    n_iter = 0
    model.save_networks(0, detcfg.OUTPUT_DIR)

    best_fooling = 0.0
    best_model_epoch = 0

    def eps_scheduler(epoch):
        if epoch <= 3:
            return 4.0 / 255.0
        else:
            return 10.0 / 255.0

    flag = True
    meters = MetricLogger()
    end = time.time()
    batch_time = time.time() - end
    meters.update(time=batch_time)

    for epoch in range(opt.epoch_count, opt.epoch_count + opt.max_epochs):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        end = time.time()
        batch_time = time.time() - end

        if epoch == 1 and opt.warm_start:
            model.eps = eps_scheduler(epoch)
            logger.info("Changing the epsilon to {:.1f}".format(model.eps * 255))

        for i, data in enumerate(dataloader):

            if 0 and total_iters % opt.print_freq == 0:
                model.update_writer(total_iters, grad=False if i == 0 else True)

            eta_seconds = meters.time.global_avg * (num_iterations_per_epoch - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if total_iters > 2000 and flag:
                model.eps = 10.0 / 255.0
                logger.info("Changing the epsilon to {:.1f}".format(model.eps * 255))
                flag = not flag

            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            model.set_input(data)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)

            if i == 0:
                logger.info(model.image_ids[:2])

            loss_dict = model.optimize_parameters()
            avg_loss_ce.update(loss_dict['ce'].item())
            avg_loss_rl.update(loss_dict['rl'].item())
            avg_loss_att.update(loss_dict['att'].item())
            avg_loss_L2.update(loss_dict['L2'].item())
            avg_loss_feat.update(loss_dict['feat'].item())

            if(niter % opt.print_freq == 0):
                logger.info("epoch: {:3d}, iter: {:4d}, l2: {:.5f}, "
                            "ce: {:.5f},  rl: {:.5f}, att: {:.5f}, feat:  {:.5f}  eta: {}".format(epoch,
                                                                                                  total_iters,
                                                                                                  avg_loss_L2.avg,
                                                                                                  avg_loss_ce.avg,
                                                                                                  avg_loss_rl.avg,
                                                                                                  avg_loss_att.avg,
                                                                                                  avg_loss_feat.avg,
                                                                                                  eta_string))
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                model.save_clean_and_adv(epoch)

            if 1 and total_iters % opt.save_latest_freq == 0:
                logger.info('saving the latest model (epoch {}, total_iters {})'.format(epoch,
                                                                                        total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix, detcfg.OUTPUT_DIR)
                model.evaluate_adv(total_iters, False)

            iter_data_time = time.time()

            if i == opt.stop_iter:
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix, detcfg.OUTPUT_DIR)
                break

            if i % 1500 == 0:
                model.save_networks('latest', detcfg.OUTPUT_DIR)

            if i == num_iterations_per_epoch:
                n_iter = 0
                break

            niter += 1

        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' %
                        (epoch, total_iters))
            model.save_networks('latest', detcfg.OUTPUT_DIR)
            model.save_networks(epoch, detcfg.OUTPUT_DIR)

        if epoch % eval_freq == 0:
            adv_accu = model.evaluate_adv(total_iters, False)

            if adv_accu['Fooling'] > best_fooling:
                best_model_epoch = epoch
                model.save_networks('best', detcfg.OUTPUT_DIR)
                best_fooling = adv_accu['Fooling']
                logger.info('saved the best model at the end of epoch: {} with fooling rate: {:.2f}%'.format(
                    best_model_epoch, 100 * best_fooling))

        logger.info('saved the best model at the end of epoch: {} with fooling rate: {:.2f}%'.format(
            best_model_epoch, 100 * best_fooling))

        logger.info("End of epoch: {:3d}, iter: {:4d}, l2: {:.5f}, "
                    "ce: {:.5f},  rl: {:.5f}, att: {:.5f}, feat:  {:.5f} eta: {}".format(epoch,
                                                                                         total_iters,
                                                                                         avg_loss_L2.avg,
                                                                                         avg_loss_ce.avg,
                                                                                         avg_loss_rl.avg,
                                                                                         avg_loss_att.avg,
                                                                                         avg_loss_feat.avg,
                                                                                         eta_string))
        model.update_learning_rate()
        torch.save({'state_dict': model.optimizer_G.state_dict()},
                   os.path.join(detcfg.OUTPUT_DIR, 'optimizer.pth'))

        if epoch == opt.max_epochs:
            logger.info("End of Training!")
            break
