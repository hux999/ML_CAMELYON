import sys
import os

import cv2
import numpy as np

sys.path.append('./ASAP/bin/')
import multiresolutionimageinterface as mir

if __name__ == '__main__':
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(sys.argv[1])
    #max_x, max_y = mr_image.getLevelDimensions(8)
    #image_patch = mr_image.getUCharPatch(0, 0, max_x ,max_y, 8)
    image_patch = mr_image.getUCharPatch(6000, 53500, 600, 600, 5)
    cv2.imshow('image_patch', image_patch)
    cv2.waitKey()

