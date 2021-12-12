import os
from options.train_options1 import TrainOptions
from models import create_model
from ssd.config import cfg as detcfg
from ssd.utils.logger import setup_logger
from ssd.utils import dist_util, mkdir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
basedir = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':

    opt = TrainOptions().parse()
    det = 'ssd320' if opt.eval_backbone == 'mobilenet_v3' else 'ssd300'
    detcfg.merge_from_file('./ssd/config/{}_{}_{}.yaml'.format(opt.eval_backbone, det, opt.eval_dataset))
    detcfg.OUTPUT_DIR = os.path.join(basedir, "..", "checkpoints", opt.generator_ckpt_dir)

    detcfg.SOLVER.BATCH_SIZE = 10
    detcfg.TEST.CONFIDENCE_THRESHOLD = 0.5
    detcfg.TEST.NMS_THRESHOLD = 0.5
    detcfg.TEST.MAX_PER_IMAGE = 100
    gen_size = 300 if 'inception' in opt.generator_ckpt_dir else 224

    detcfg.freeze()

    if detcfg.OUTPUT_DIR:
        mkdir(detcfg.OUTPUT_DIR)

    opt.detcfg = detcfg
    opt.eps = 16.0 / 255.0
    opt.perturbmode = False
    opt.isTrain = False

    # build generator and detector models
    model = create_model(opt)
    model.setup(opt)

    logger = setup_logger("SSD", dist_util.get_rank(), opt.detcfg.OUTPUT_DIR, opt.logfile)
    logger.info("Detection model path: {}".format(opt.detector_ckpt))
    logger.info("Generator model path: {}/{}_net_G.pth".format(detcfg.OUTPUT_DIR, opt.generator_ckpt_epoch))

    opt.num_images = -1

    # run the models
    model.evaluate(iteration=0, save_feats=False)
    model.evaluate_adv(iteration=0, save_feats=False, gen_size=gen_size)
