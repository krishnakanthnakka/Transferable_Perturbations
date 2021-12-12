from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset
from .clipart import CLIPARTDataset
from .comic import COMICDataset
from .watercolor import WATERCOLORDataset
from .cityscapes import CITYSCAPESDataset
from .sim10k import SIM10KDataset
from .foggycityscapes import FOGGYCITYSCAPESDataset


_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'CLIPARTDataset': CLIPARTDataset,
    'COMICDataset': COMICDataset,
    'WATERCOLORDataset': WATERCOLORDataset,
    "CITYSCAPESDataset": CITYSCAPESDataset,
    "SIM10KDataset": SIM10KDataset,
    "FOGGYCITYSCAPESDataset": FOGGYCITYSCAPESDataset
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset:
            args['remove_empty'] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
