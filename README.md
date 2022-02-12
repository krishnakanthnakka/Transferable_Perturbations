
## Introduction

This is an official release of the paper **Learning Transferable Adversarial Perturbations**. The code is in early release and will be updated soon


## Installation

1. It is tested with the following  packages:

    ``` text
      PyTorch: 1.7.1+cu101
      Python: 3.6.9
      Torchvision: 0.8.2+cu101
      CUDA: 10.1
      CUDNN: 7603
      NumPy: 1.18.1
      PIL: 7.0.0
   ```

2. Download source code from GitHub
   ```
    git clone https://github.com/krishnakanthnakka/Transferable_Perturbations.git
   ```
3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name LTP python=3.6.9
   ```
4. Activate conda environment
   ```
    source activate LTP
   ```
5. Install requirements
   ```
    pip install -r requirements.txt
    ```
6. Download pretrained generator checkpoints from our model zoo ([GoogleDrive](https://drive.google.com/drive/folders/1QkJh9EPGyq_LnzzU5mzpkBNhJFxIxGMu?usp=sharing)) and place them in the root folder



### Data preparation

1. The data structure of ```ImageNet``` looks like below:

    ```text
    /datasets/
    ├── imagenet
    │   ├── train
    │   │   ├── n02328150
    │   │   ├── n03447447
    │   ├── val
    │   │   ├── n02328150
    │   │   ├── n03447447
    |-- comics
    |   |-- train
    |   |   |── All-Star Section Eight
    |   |   |── Action Comics v2

    ```

2. We evaluate on 5K random samples from ImageNet val-set. We release the 5K samples in text file at ```cda/data/datasets/imagenet5k_val.txt```




### Training

1. To train generator against squeezenet classfier pretrained on imagenet
   ```bash
   bash scripts/imagenet/squeezenet.sh
   ```

2. To train generator against onother datasets, prepare the config file at ```cda/config/``` and put the images file at ```cda/data/datasets/```


## Testing on Cross-Model Setting

1. Set all environmental paths and other packages in path by ```source envs.sh```

2. For attacking PyTorch Image recognition models using the generator trained on ```squeezenet``` discriminator and ```imagenet``` dataset:
   ```bash
   bash run_exps.sh  squeezenet1_1 imagenet feat
   ```


### Results on ImageNet models

1. We report fooling rate metric (percentage of images for which label is flipped) on ImageNet5K val-set.
    | Train  | VGG16 | ResNet152 | Inceptionv3 | DenseNet121 | SqueezeNet1.1 | ShuffleNet  | MNASNet  |    MobileNet |
    | :---:  | :---: | :---:     | :---:       | :---:       | :---:      | :---:       |  :---:   |       :---:  |
    |  VGG16| 99.32% |68.38%    | 46.60%        |84.68%      | 86.52%     | 67.84%      | 90.44%   |   60.08%     |
    |SqueezeNet1.1|  96.06%|   76.44%|   70.66% |   88.70%  |  99.68% |   90.06%        | 90.06%    |   84.58%|


## Testing on Cross-Task Setting on SSD


1. To run SSD experiments, first enter the  ```SSD``` folder and set paths to SSD library
   ```bash
   cd SSD
   source envs.sh
   ```
2. Download and place the trained SSD models from [GoogleDrive](https://drive.google.com/drive/folders/13TLIHLjDh4IeSiA5vXIqnLpCOwNdzxI9?usp=sharing) and place in this SSD folder.
   We used publicly available [SSD](https://github.com/lufficc/SSD) implementation to train models for 120K iterations.

3. Prepare the VOC dataset in ```datasets/2007``` and ```datasets/2012```

    ```text
        /SSD/datasets/
        ├── VOC2007
        │   ├── JPEGImages
        │   ├── ImageSets
        │   ├── Annotations
        │   ├── .....
        ├── VOC2012
        │   ├── JPEGImages
        │   ├── ImageSets
        │   ├── Annotations
        │   ├── .....

4. For attacking SSD models using the generator trained on ```squeezenet``` discriminator and ```imagenet``` dataset:
   ```bash
   bash run_exps.sh  squeezenet1_1 imagenet feat
   ```

### Results on SSD detectors

1. We report mAP on PASCAL VOC test set before and after attack with generator trained against SqueezeNet discriminator and ImageNet data.

    | Train  | VGG16 | ResNet50 | EfficientNet | MobileNet |
    | :---:  | :---: | :---:     | :---:       | :---:     |
    |Clean|  68.1|   66.1|   61.1|   55.4 |
    |VGG16|  8.30|   11.8|   11.4 |   9.12 |
    |SqueezeNet1.1|  13.1|   10.8|   11.5 |   6.19 |


## Citation

```bibtex
@inproceedings{nakka2021learning,
    title={Learning Transferable Adversarial Perturbations},
    author={Krishna Kanth Nakka and Mathieu Salzmann},
    year={2021},
    booktitle={NeurIPS},
}
```
