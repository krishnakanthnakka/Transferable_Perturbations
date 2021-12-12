import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {

        'watercolor_train': {
            "data_dir": "watercolor",
            "split": "train"
        },

        'watercolor_test': {
            "data_dir": "watercolor",
            "split": "test"
        },

        'comic_train': {
            "data_dir": "comic",
            "split": "train"
        },

        'comic_test': {
            "data_dir": "comic",
            "split": "test"
        },
        'clipart_train': {
            "data_dir": "clipart",
            "split": "train"
        },

        'clipart_test': {
            "data_dir": "clipart",
            "split": "test"
        },

        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },

        'voc_200712_trainval_ood': {
            "data_dir": "VOC200712_OOD",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        }, "cityscapes_trainval": {
            "data_dir": "Cityscapes",
            "split": "trainval"
        },

        "cityscapes_test": {
            "data_dir": "Cityscapes",
            "split": "test"
        },

        "foggycityscapes_trainval": {
            "data_dir": "FoggyCityscapes",
            "split": "trainval"
        },

        "foggycityscapes_test": {
            "data_dir": "FoggyCityscapes",
            "split": "test"
        },

        "sim10k_train": {
            "data_dir": "SIM10K",
            "split": "trainval8k_caronly"
        },
        "sim10k_test": {
            "data_dir": "SIM10K",
            "split": "test2k_caronly"
        }
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        elif "clipart" in name:
            clipart_root = DatasetCatalog.DATA_DIR
            # if 'VOC_ROOT' in os.environ:
            #     voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(clipart_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="CLIPARTDataset", args=args)

        elif "comic" in name:
            comic_root = DatasetCatalog.DATA_DIR
            # if 'VOC_ROOT' in os.environ:
            #     voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(comic_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="COMICDataset", args=args)

        elif "watercolor" in name:
            watercolor_root = DatasetCatalog.DATA_DIR
            # if 'VOC_ROOT' in os.environ:
            #     voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(watercolor_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="WATERCOLORDataset", args=args)

        elif "sim10k" in name:
            sim10k_root = DatasetCatalog.DATA_DIR
            # if 'VOC_ROOT' in os.environ:
            #     voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(sim10k_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="SIM10KDataset", args=args)

        elif "cityscapes" in name:
            city_root = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(city_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="CITYSCAPESDataset", args=args)

        elif "foggycityscapes" in name:
            city_root = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(city_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="FOGGYCITYSCAPESDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
