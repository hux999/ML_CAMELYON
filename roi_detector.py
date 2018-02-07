import sys
import os
import random

import cv2
import numpy as np

sys.path.append('./ASAP/bin/')
import multiresolutionimageinterface as mir

cache_path = './cache/roi_cache'

def is_empty(image_patch):
    max_val = np.max(np.max(image_patch,axis=0), axis=0)
    min_val = np.min(np.min(image_patch,axis=0), axis=0)
    return np.max(max_val-min_val) < 8 

def foreground_mask(image_patch, blob_size):
    stride = int(blob_size/2)
    max_y, max_x, _ = image_patch.shape
    fg_mask = np.ones((max_y,max_x), np.uint8)*255
    #image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    for y in range(0, max_y-stride, stride):
        for x in range(0, max_x-stride, stride):
            if is_empty(image_patch[y:y+blob_size, x:x+blob_size, :]):
                fg_mask[y:y+blob_size,x:x+blob_size] = 0
    return fg_mask

class RoiDetector:
    def __init__(self, tif_file, base_level=5, blob_size=20):
        self.tif_file = tif_file
        self.base_level = base_level
        self.scale = 2**base_level
        self.blob_size = blob_size
        cache_file = os.path.join(cache_path, self.tif_file.split('/')[-1][:-4]+'.npy')
        if os.path.exists(cache_file):
            print('load cache file: %s' % cache_file)
            self.fg_mask = np.load(cache_file)
        else:
            self.fg_mask = self.get_mask()
            np.save(cache_file[:-4], self.fg_mask)
        self.idx_y, self.idx_x = self.get_indice()

    def get_mask(self):
        reader = mir.MultiResolutionImageReader()
        level = self.base_level
        while level<=9:
            mr_image = reader.open(self.tif_file)
            max_x, max_y = mr_image.getLevelDimensions(level)
            image_patch = mr_image.getUCharPatch(0, 0, max_x, max_y, level)
            if not is_empty(image_patch):
                break
            level += 2
        assert(level<=9)
        self.mask_level = level
        fg_mask = foreground_mask(image_patch, self.blob_size)
        if level != self.base_level:
            scale = (level-self.base_level)**2
            fg_mask = cv2.resize(fg_mask, None, fx=scale, fy=scale)
        return fg_mask

    def get_indice(self):
        idx_y, idx_x = np.where(self.fg_mask>0)
        idx_y = idx_y * self.scale + self.scale/2
        idx_x = idx_x * self.scale + self.scale/2
        return idx_y, idx_x

    def is_foreground(self, x, y):
        x = int(x/self.scale)
        y = int(y/self.scale)
        return self.fg_mask[y, x] > 0

    def random_choice(self):
        select = random.randint(0, self.idx_x.shape[0]-1)
        return self.idx_y[select], self.idx_x[select]

if __name__ == '__main__':
    tif_root = sys.argv[1]
    for tif_file in os.listdir(tif_root): 
        tif_file = os.path.join(tif_root, tif_file)
        print(tif_file)
        level = 5
        reader = mir.MultiResolutionImageReader()
        fg_roi = RoiDetector(tif_file, level)

        mr_image = reader.open(tif_file)
        for i in range(10):
            print(i, mr_image.getLevelDownsample(i))
        max_x, max_y = mr_image.getLevelDimensions(fg_roi.mask_level)
        image_patch = mr_image.getUCharPatch(0, 0, max_x, max_y, fg_roi.mask_level)

        fg_mask = fg_roi.fg_mask

        print('level', fg_roi.mask_level)

        scale = 1.0/8
        image_patch = cv2.resize(image_patch, None, fx=scale, fy=scale)
        cv2.imshow('image_patch', image_patch)

        fg_mask = cv2.resize(fg_mask, None, fx=scale, fy=scale)
        fg_mask = cv2.merge([fg_mask, fg_mask, fg_mask])
        for i in range(1000):
            y, x = fg_roi.random_choice()
            cv2.circle(fg_mask, (x/256,y/256), 2, (0,255,0))
        cv2.imshow('fg_mask', fg_mask)
        key = cv2.waitKey()
