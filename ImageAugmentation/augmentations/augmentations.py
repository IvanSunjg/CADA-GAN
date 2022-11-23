import numpy as np
from . import utils
from torchvision import transforms


class IndexTransform():

    '''
    An abstract class that is used to indicate when a transform is to be transformed on an item,
    i.e. an image-target pair. Used by transforms including, but not limited to, MixUp and CutMix.
    '''

    def __call__(self, img):
        raise NotImplementedError('Don not call this Class directly!')

def apply_transform(t, img, idx=None):

    if isinstance(t, IndexTransform):
        img = t(img, idx)
    else:
        img = t(img)

    return img

class _Mix():

    '''
    Superclass for transformations that require image mixing of the mother and father.
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def get_mix_item(self, idx):

        # If the image target is dad
        if idx >=456 or idx < 684:
            mix_idx = idx + 228
        
        # If the image target is mother
        elif idx >= 684:
            mix_idx = idx - 228
        
        else:
            raise ValueError(f'index you pass to Mix class is wrong! Please make sure the index of the image belongs to either Mother or Father!')
        
        mix_image, mix_label = self.dataset[mix_idx]

        return mix_image

class MixUp(_Mix, IndexTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, idx):

        mix_image = super().get_mix_item(idx=idx)
        img = utils.mixup(img, mix_image, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'