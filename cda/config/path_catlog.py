import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        "imagenet_val": {
            "data_dir": "imagenet",
            "split": "val"
        },

        "imagenet5k_val": {
            "images_path": "imagenet5k_val.txt",
            "data_dir": "imagenet",
            "split": "val"
        },


    }

    @staticmethod
    def get(name):
        if ("imagenet" in name) or ("imagenetnips1k_val") or ("imagenet5k" in name) or ("imagenet1k" in name) \
            or ("imagenet20k" in name) or ('chestxsmall' in name) or ('chestxfull' in name) or ('comics' in name)\
            or ('comics10k' in name) or ('paintings10k' in name) or ('paintings' in name) or ('photo' in name) \
                or ('cartoon' in name) or ('art_painting' in name) or ('sketch' in name) or ('pacs' in name):

            voc_root = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(data_dir=os.path.join(
                voc_root, attrs["data_dir"]), images_path=attrs['images_path'], split=attrs["split"],)
            return dict(factory="ImageDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
