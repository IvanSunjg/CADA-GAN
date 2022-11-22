# AUgmentation

A python library that provides image augmentations for the Deep Learning Project of ETHz 2022. AUgmentations implements a modified version of the pytorch data loading and augmentation pipeline that allows for transformation of both the sample and target simultaneously. The three main blocks that can be directly plugged in are MixUp, AugMix, and SmartAugmentation, which are used specifically for augmenting the dataset of the Child Face Generation GAN.

## Installation

The library can be installed from GitHub using `pip`.

For Linux and MacOS:

```bash
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations&subdirectory=avgmentations'
```

For Windows:

```bash
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations^&subdirectory=avgmentations'
```

## Getting Started
