import numpy as np
from . import utils


class IndexTransform():

    '''
    An abstract class that is used to indicate when a transform is to be transformed on an item,
    i.e. an image-target pair. Used by transforms including, but not limited to, MixUp and CutMix.
    '''

    def __call__(self, img):
        raise NotImplementedError('Don not call this Class directly!')

def apply_transform(ts, img, idx=None):

    for t in ts:

        if isinstance(t, IndexTransform):
            img = t(img, idx)
        else:
            img = t(img)

    return img

class _Mix():

    '''
    Superclass for transformations that require image mixing of the mother and father.
    '''
    def get_mix_item(self, idx, n):

        # If the image target is dad
        if idx >=0 and idx < n:
            mix_idx = idx + n
        
        # If the image target is mother
        elif idx >= n and idx < 2 * n:
            mix_idx = idx - n

        mix_image, mix_label = self.dataset[mix_idx]

        return mix_image

    def __init__(self, dataset):
        self.dataset = dataset

class MixUp_FMD(_Mix, IndexTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, idx):

        if idx < 274 * 2:
            mix_image = super().get_mix_item(idx=idx,n=274)
            img = utils.mixup(img, mix_image, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'


class MixUp_FMS(_Mix, IndexTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, idx):

        if idx < 285 * 2:
            mix_image = super().get_mix_item(idx=idx,n=285)
            img = utils.mixup(img, mix_image, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'


class MixUp_FMSD(_Mix, IndexTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, idx):

        if idx < 228 * 2:
            mix_image = super().get_mix_item(idx=idx,n=228)
            img = utils.mixup(img, mix_image, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'

class P(IndexTransform):

    '''
    Apply a transformation with a probability.
    '''

    def __init__(self, transform, p):
        self.transform = transform
        self.p = p

    def __call__(self, img, idx):
        if np.random.rand() < self.p:
            return apply_transform([self.transform], img, idx)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'({self.transform}, {self.p})'

class AugMix():
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augumentations taken (default 3)
    w1,w2,w3: weight for each augumentated image to mixup
    m: weight for mix with the original and the mixup augumentated image
    level: level of augmentation
    '''

    def __init__(self, k=3, w=[0.2, 0.3, 0.5], m=0.2, level=3):
        self.k = k
        self.w = w
        self.m = m
        self.level = level

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        '''

        miximg = utils.augmix(img, k=self.k, w=self.w, m=self.m, level=self.level)
        return miximg

    def __repr__(self):
        return self.__class__.__name__ + f'(k={self.k}, w={self.w}, m={self.m}, level={self.level})'