# AUgmentation

A python library that provides image augmentations for the Deep Learning Project of ETHz 2022. AUgmentations implements a modified version of the pytorch data loading and augmentation pipeline that allows for transformation of both the sample and target simultaneously. The three main blocks that can be directly plugged in are MixUp, AugMix, and SmartAugmentation, which are used specifically for augmenting the dataset of the Child Face Generation GAN.

## Dependencies

* torch - pytorch-gpu (or cpu)
* torchvision
* numpy
* matplotlib

Please make sure you install all the packages above before you install the **AU**gmentation Library.

## Installation

The library can be installed from GitHub using `pip`.

For Linux and MacOS:

```bash
pip install 'git+https://github.com/IvanSunjg/ETH_DL_2022.git#egg=augmentations&subdirectory=augmentations'
```

For Windows:

```bash
pip install 'git+https://github.com/IvanSunjg/ETH_DL_2022.git#egg=augmentations^&subdirectory=augmentations'
```

## Getting Started

