import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from cda.data import samplers
from cda.data.datasets import build_dataset
from cda.data.transforms import build_transforms, build_target_transform
from cda.structures.container import Container
from colorama import Fore, Style
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random  # CHANGED


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])

        targets = default_collate(transposed_batch[1])
        img_ids = default_collate(transposed_batch[2])

        # if self.is_train:
        #     list_targets = transposed_batch[1]
        #     targets = Container(
        #         {key: default_collate([d[key] for d in list_targets])
        #          for key in list_targets[0]}
        #     )
        # else:
        #     targets = None
        return images, targets, img_ids


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# CHANGED
# def worker_init_fn(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0, shuffle=None, data_aug=None, data_aug_transforms=False):
    logger = logging.getLogger("CDA.inference")
    logger.info("Start preparing dataloaders!")

    train_transform = build_transforms(
        cfg, is_train=is_train, data_aug_transforms=data_aug_transforms)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, transform=train_transform,
                             target_transform=target_transform, is_train=is_train, data_aug=data_aug)

    if shuffle is None:
        shuffle = is_train

    logger.info("Shuffle: {}".format(shuffle))
    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(
                batch_sampler, num_iterations=max_iter, start_iter=start_iter)


        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train),
                                 worker_init_fn=worker_init_fn,  # CHANGED
                                 )


        data_loaders.append(data_loader)
        # logger.info("Dataset: {}, {}size:{} {}".format(
        #     data_loader.dataset, Fore.RED, len(dataset), Style.RESET_ALL))

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
