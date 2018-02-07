import random
import numpy as np 
import cv2

class ReColor:
    def __init__(self, alpha=0.05, beta=0.2):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, im):
        # random amplify each channel
        t = np.random.uniform(-1, 1, 3)
        im = im.astype(np.float32)
        im *= (1 + t * self._alpha)
        mx = 255. * (1 + self._alpha)
        up = np.random.uniform(-1, 1)
        im = np.power(im / mx, 1. + up * self._beta)
        im = im * 255
        return im

class RandomJitter:
    def __init__(self, max_angle=180, max_scale=0.1):
        self._max_angle = max_angle
        self._max_scale = max_scale

    def __call__(self, im, mask):
        h,w,_ = im.shape
        center = w/2.0, h/2.0
        angle = np.random.uniform(-self._max_angle, self._max_angle)
        scale = np.random.uniform(-self._max_scale, self._max_scale) + 1.0
        m = cv2.getRotationMatrix2D(center, angle, scale)
        im = cv2.warpAffine(im, m, (w,h))
        mask = cv2.warpAffine(mask, m, (w,h))
        return im, mask

class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, im, mask):
        if random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        return im, mask

class RandomRotate:
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, im, mask):
        rotate = random.randint(0, 3)
        if self.random_flip and random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        if rotate > 0:
            im = np.rot90(im, rotate)
            mask = np.rot90(mask, rotate)
        return im.copy(), mask.copy()
