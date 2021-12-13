import errno
import os
import datetime
from shutil import copy


def str2bool(s):
    return s.lower() in ('true', '1')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def savefiles(save_dir, opt):
    dst_folder = os.path.join(save_dir,
                              "files_{}".format(datetime.datetime.now().strftime("%H_%M_%S")))

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # root_dir = save_dir.split("/")[:-2].join("/")

    root_dir = '/'.join([x for x in save_dir.split('/')[:-2]])
    copy(os.path.join(root_dir, "train_gen3.py"), dst_folder)

    copy(os.path.join(root_dir, "train_gen.py"), dst_folder)
    copy(os.path.join(root_dir, "pix2pix/models",
                      "Generator_model.py"), dst_folder)

    copy(os.path.join(root_dir, "pix2pix/models",
                      "resnet_gen.py"), dst_folder)

    output_file_name = "/../{}_{}_{}_run_{}.yaml".format(opt.train_dataset,
                                                     opt.train_classifier, opt.loss_type, str(opt.run))

    config_file = "{}_{}.yaml".format(opt.train_classifier, opt.train_dataset)

    copy(os.path.join(root_dir, "cda/config/", config_file),
         dst_folder + output_file_name)
    copy(os.path.join(root_dir, "cda/modeling/detector/loss.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/at.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/resnet.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/squeezenet.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/inception.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/vgg.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/modeling/detector/densenet.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/data/transforms/__init__.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/data/datasets/image.py"), dst_folder)
    copy(os.path.join(root_dir, "cda/data/build.py"), dst_folder)
