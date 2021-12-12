
## Introduction

This is an official release of the paper **Learning Transferable Adversarial Perturbations**. The code is in early release and will be updated soon



## Installation

It requires the following OpenMMLab packages:

- PyTorch: 1.7.1+cu101
- Python: 3.6.9
- Torchvision: 0.8.2+cu101
- CUDA: 10.1
- CUDNN: 7603
- NumPy: 1.18.1
- PIL: 7.0.0

1. Download source code from GitHub
   ```
    git clone https://github.com/krishnakanthnakka/Transferable_Perturbations.git
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name LTP python=3.6.9
   ```
3. Activate conda environment
   ```
    source activate LTP
   ```
4. Install requirements
   ```
    pip install -r requirements.txt
    ```
5. Download pretrained generator checkpoints from our model zoo ([GoogleDrive](https://drive.google.com/drive/folders/1QkJh9EPGyq_LnzzU5mzpkBNhJFxIxGMu?usp=sharing)) and place them in the root folder



### Data preparation

1. The data structure of ```ImageNet1M``` looks like below:

    ```text
    /path/to/ImageNet/
    ├── ImageNet1M
    │   ├── train
    │   │   ├── n02328150
    │   │   ├── n03447447
    │   ├── val
    │   │   ├── n02328150
    │   │   ├── n03447447
    ```


## Testing on Cross-Model Setting


1.We report fooling rate metric (percentage of images for which label is flipped) on ImageNet5K val-set.


| Train  | VGG16 | ResNet152 | Inceptionv3 | DenseNet121 | SqueezeNet1.1 | ShuffleNet  | MNASNet  |    MobileNet |
| :---:  | :---: | :---:     | :---:       | :---:       | :---:      | :---:       |  :---:   |       :---:  |
|  VGG16| 99.32% |68.38%    | 46.60%        |84.68%      | 86.52%     | 67.84%      | 90.44%   |   60.08%     |
|ResNet152|99.10%|  99.72%  | 74.90%        |  98.82%   | 89.12%        | 96.48%    | 94.00%    |86.44%         |
|SqueezeNet1.1|  98.52%|   86.67%|   75.54% |   93.57%  |  92.47% |   89.44%        | 92.91%    |   82.75%|


## Testing on Cross-Task Setting on SSD


1. To run SSD experiments, first enter the  ```SSD``` folder
   ```bash
   cd SSD
   ```
2. Download and place the trained SSD models from [GoogleDrive](https://drive.google.com/drive/folders/13TLIHLjDh4IeSiA5vXIqnLpCOwNdzxI9?usp=sharing) and place in this SSD folder.
   We used publicly available [SSD](https://github.com/lufficc/SSD) implementation to train models for 120K iterations.

3. Prepare the VOC dataset in ```datasets/2007``` and ```datasets/2012```

4. For attacking SSD models using the generator trained on squeezenet discriminator and imagenet dataset:
   ```bash
   bash run_exps.sh  squeezenet1_1 imagenet feat
   ```

### Results on SSD detectors

1. We report mAP on PASCAL VOC test set before and after attack with generator trained against SqueezeNet discriminator and ImageNet data.

    | Train  | VGG16 | ResNet50 | EfficientNet | MobileNet |
    | :---:  | :---: | :---:     | :---:       | :---:     |
    |Clean|  68.1|   66.1|   61.1|   55.4 |
    |SqueezeNet1.1|  13.1|   10.8|   11.5 |   6.19 |


## ToDO

- Continue updating the repository




## Citation

```bibtex
@inproceedings{nakka2021learning,
    title={Learning Transferable Adversarial Perturbations},
    author={Krishna Kanth Nakka and Mathieu Salzmann},
    year={2021},
    booktitle={NeurIPS},
}
```
