
## Learning Transferable Adversarial Perturbations

This is an official release of the paper **Learning Transferable Adversarial Perturbations**. I'm going through unfortunate medical emergency of my wife. It will be updated by December 12 week-end. Thank you all for understanding.



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


### Data preparation

The data structure of ```ImageNet1M``` looks like below:

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


### Results on ImageNet5K

| Train  | VGG16 | ResNet152 | Inceptionv3 | DenseNet121 | SqueezeNet1.1 | ShuffleNet  | MNASNet  |    MobileNet |
| :---:  | :---: | :---:     | :---:       | :---:       | :---:      | :---:       |  :---:   |       :---:  |
|  VGG16| 99.32% |68.38%    | 46.60%        |84.68%      | 86.52%     | 67.84%      | 90.44%   |   60.08%     |
|ResNet152|99.10%|  99.72%  | 74.90%        |  98.82%   | 89.12%        | 96.48%    | 94.00%    |86.44%         |
|SqueezeNet1.1|  98.52%|   86.67%|   75.54% |   93.57%  |  92.47% |   89.44%        | 92.91%    |   82.75%|

## Citation

```bibtex
@inproceedings{nakka2021learning,
    title={Learning Transferable Adversarial Perturbations},
    author={Krishna Kanth Nakka and Mathieu Salzmann},
    year={2021},
    booktitle={NeurIPS},
}
```
