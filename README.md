# Computer Vision Lab. Image Super Resolution Task 2023
## Contents
- [Introduction](#introduction)
- [Data](#data)
  - [Training Data](#training-data)
  - [Test Data](#test-data)
- [Architectures](#architectures)
  - [MDRN](#mdrn)
  - [GFMN](#gfmn)
- [Demo Results](#demo)
- [Experiments](#experiments)
  - [Addition tasks](#dddition-tasks)
  - [Losses modification](#losses-modification)
- [Setup and Reproducibility](#setup-and-reproducibility)
  - [Environment](#environment)
  - [Reproducibility](#reproducibility)
- [Links](#links)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction
A study of the result of the Image Super Resolution NTIRE 2023 competition. Study and comparison of MDRN and GFMN models. Experimental improvement of models quality.
## Data
### Training data
Data from the DIV2K and LSDIR public datasets are used to train the models. DIV2K dataset consists of 1,000 diverse 2K resolution RGB images, which are split into a training set of 800 images, a validation set of 100 images,and a test set of 100 images. LSDIR dataset contains 86,991 high-resolution high-quality images, which are split into a training set of 84,991 images, a validation set of 1,000 images, and a test set of 1,000.
### Test data
There are 4 datasets used to test the quality of models: DIV2K test dataset, Manga109 containing comics images, publicly available dataset for Super Resolution quality test Set5 and custom dataset Christmas.
![fig](README.assets/datasets.png)
# Architectures
### **MDRN**
**Multi-level Dispersion Residual Network (MDRN).** NTIRE 2023 winner under:
- Number of parameters.
- FLOPS.
![fig](README.assets/architecture.png)
![fig](README.assets/EADB_details.png)
**Gated feature modulation network (GFMN).** NTIRE 2023 3rd place under:
- Number of parameters.
- FLOPS.
![fig](README.assets/architecture.png)
![fig](README.assets/EADB_details.png)

## 💻Environment

- [PyTorch >= 1.9](https://pytorch.org/)
- [Python 3.7](https://www.python.org/downloads/)
- [Numpy](https://numpy.org/)
- [BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

## 🔧Installation

```python
pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```

## 📜Data Preparation

The trainset uses the DIV2K (800) + LSDIR(the first 10k). In order to effectively improve the training speed, images are cropped to 480 * 480 images by running script extract_subimages.py, and the dataloader will further randomly crop the images to the GT_size required for training. GT_size defaults to 128/192/256 (×2/×3/×4). 

```python
python extract_subimages.py
```

The input and output paths of cropped pictures can be modify in this script. Default location: ./datasets/DL2K.

## 🚀Train

▶️ You can change the training strategy by modifying the configuration file. The default configuration files are included in ./options/train/MDRN. Take one GPU as the example.

```python
### Train ###
### MDRN ###
python train.py -opt ./options/train/MDRN/train_mdrn_x2.yml --auto_resume  # ×2
python train.py -opt ./options/train/MDRN/train_mdrn_x3.yml --auto_resume  # ×3
python train.py -opt ./options/train/MDRN/train_mdrn_x4.yml --auto_resume  # ×4
```

For more training commands, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)

## :toilet:Test

▶️ You can modify the configuration file about the test, which is located in ./options/test/MDRN. At the same time, you can change the benchmark datasets and modify the path of the pre-train model. 

▶️ We provide all MDRN and MDRN-S (for NTIRE2023 ESR) pre-trained models, located in the folder ./pretrain_models.

▶️ All benchmark datasets can be obtained from the official website.  You should update the paths in the configuration files based on the paths where benchmark datasets are located on your computer.

```python
### Test ###
### MDRN ###
python basicsr/test.py -opt ./options/test/MDRN/test_mdrn_x2.yml  # ×2
python basicsr/test.py -opt ./options/test/MDRN/test_mdrn_x3.yml  # ×3
python basicsr/test.py -opt ./options/test/MDRN/test_mdrn_x4.yml  # ×4
```

## 🚩Results


## :rainbow:References

```bibtex
@InProceedings{Mao_2023_CVPR,
    author    = {Mao, Yanyu and Zhang, Nihao and Wang, Qian and Bai, Bendu and Bai, Wanying and Fang, Haonan and Liu, Peng and Li, Mingyue and Yan, Shengbo},
    title     = {Multi-Level Dispersion Residual Network for Efficient Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1660-1669}
}
```

