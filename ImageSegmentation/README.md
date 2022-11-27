# Image Segmentation

A code base that applies image segmentation in the context of face parsing. The code is from "Lin et al. 2021 - RoI Tanh-polar transformer network for face parsing in the wild" (https://github.com/hhj1897/face_parsing).

## Dependencies

* git-lfs
* numpy >= 1.17.0
* opencv-python >= 3.4.2
* torch >= 1.6.0
* torchvision >= 0.7.0
* scipy >= 1.1.0

## Installation

```bash
git clone https://github.com/IvanSunjg/ETH_DL_2022/tree/main/ImageSegmentation
cd face_parsing
git lfs pull
pip install -e .
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ../
git clone https://github.com/hhj1897/roi_tanh_warping
cd roi_tanh_warping
pip install -e .
cd ../
```

## Getting Started

Test
```bash
python face_parsing_test.py
```

Command line arguments
```bash
--input: Path to input images
--output: Path to save images
--facebox: add rectangle around detected faces
--blurring: apply segmentation only (0), segmentation + blurring (1), blurring only (2)
```
