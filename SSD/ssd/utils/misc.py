import errno
import os
from shutil import copy
import datetime


def str2bool(s):
    return s.lower() in ('true', '1')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def savefiles(save_dir):
    dst_folder = os.path.join(save_dir,
                              "files_{}".format(datetime.datetime.now().strftime("%H_%M_%S")))

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # root_dir = save_dir.split("/")[:-2].join("/")

    root_dir = '/'.join([x for x in save_dir.split('/')[:-2]])

    copy(os.path.join(root_dir, "train_gen.py"), dst_folder)
    copy(os.path.join(root_dir, "pix2pix/models",
                      "G_search_L2_500_regress_model.py"), dst_folder)
