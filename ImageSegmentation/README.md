# Image Segmentation

A code base that applies image segmentation in the context of face parsing. The code is from "Lin et al. 2021 - RoI Tanh-polar transformer network for face parsing in the wild" (https://github.com/hhj1897/face_parsing).

## Dependencies

* git-lfs
* numpy
* opencv-python
* torch
* torchvision

## Installation

```bash
git clone https://github.com/hhj1897/face_parsing
cd face_parsing
git lfs pull
pip install -e .
```

## Getting Started

Test
```bash
python face_warping_test.py -i 0 -e rtnet50 --decoder fcn -n 11 -d cuda:0
```

Command line arguments
```bash
-i VIDEO: Index of the webcam to use (start from 0) or
          path of the input video file
-d: Device to be used by PyTorch (default=cuda:0)
-e: Encoder (default=rtnet50)
--decoder: Decoder (default=fcn)
-n: Number of facial classes, can be 11 or 14 for now (default=11)
```
