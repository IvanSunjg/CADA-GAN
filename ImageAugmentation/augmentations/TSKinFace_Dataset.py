from torchvision.datasets import ImageFolder
from . import augmentations
import numpy as np
import sys
import os

class TSKinDataset(ImageFolder):

    def __init__(self, root, epoch_transforms={}, **kwargs):
        super(TSKinDataset, self).__init__(root, **kwargs)

        self.epoch_transforms = epoch_transforms
        self.classes, self.class_to_idx = self._find_classes(root)

        if 0 in self.epoch_transforms:
            self.transform = self.epoch_transforms[0]
    
    def _find_classes(self, root):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)

        if self.transform is not None:
            sample = augmentations.apply_transform(self.transform, sample, idx)
        return sample, target

    def update_transform(self, epoch):
        if epoch in self.epoch_transforms:
            self.transform = self.epoch_transforms[epoch]

    