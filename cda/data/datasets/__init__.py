from torch.utils.data import ConcatDataset
from cda.config.path_catlog import DatasetCatalog
from .image import ImageDataset


_DATASETS = {
    'ImageDataset': ImageDataset,
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True, data_aug=None):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        args['convolve_image'] = is_train

        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform

        args['is_train'] = is_train
        args['data_aug'] = data_aug

        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
