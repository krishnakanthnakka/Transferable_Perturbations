import os, sys
import logging
import warnings
from options.train_options1 import TrainOptions
from models import create_model
from utils import create_logger
from cda.config import cfg as detcfg
from cda.utils.logger import setup_logger
from cda.utils import dist_util, mkdir, savefiles

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
basedir = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':

    command_line = 'python ' + ' '.join(sys.argv)
    opt = TrainOptions().parse()
    opt.run = opt.train_config
    opt.gen_dropout = 0.0


    opt.loss_fn, opt.softmax2D = feat_loss_mutliscale_fn, False
    opt.act_layer_mean = False
    opt.data_dim = 'high'
    detcfg.merge_from_file('./checkpoints/{}/{}.yaml'.format(opt.train_config, opt.train_config))
    detcfg.TEST.BATCH_SIZE = 5
    detcfg.OUTPUT_DIR = os.path.join(basedir, "checkpoints", str(opt.train_config))
    detcfg.EVAL_MODEL.META_ARCHITECTURE = opt.eval_model
    detcfg.EVAL_MODEL.NUM_CLASSES = opt.eval_num_classes
    detcfg.DATASETS.TEST = (opt.eval_dataset,)
    opt.isTrain = False
    opt.classifier_weights = ''
    detcfg.MODEL.BACKBONE.PRETRAINED = True
    detcfg.MODEL.NUM_CLASSES = 1000


    # we evaluate on the resolution on which GAN is trained against
    if 'inception' in opt.train_config:
        detcfg.INPUT.RESIZE_SIZE = 300
        detcfg.INPUT.IMAGE_SIZE = 299
        opt.train_getG_299 = True
    else:
        detcfg.INPUT.RESIZE_SIZE = 256
        detcfg.INPUT.IMAGE_SIZE = 224
        opt.train_getG_299 = False


    detcfg.freeze()

    if detcfg.OUTPUT_DIR:
        mkdir(detcfg.OUTPUT_DIR)

    opt.detcfg = detcfg
    opt.weightfile = ''
    opt.continue_train = True
    opt.eps = opt.eps / 255.0
    opt.perturbmode = detcfg.NETG.PERTURB_MODE


    logger = setup_logger("CDA", dist_util.get_rank(), opt.detcfg.OUTPUT_DIR, logger_name='log_eval.txt')
    logger.info("Command line: {}".format(command_line))
    logger = logging.getLogger("CDA.inference")
    logger.info("Epsilon: {:.1f}".format(opt.eps * 255))
    model = create_model(opt)
    model.setup(opt)


    save_feats = False
    model.evaluate(total_iters, save_feats)
    model.evaluate_adv(total_iters, save_feats)
