import sys
import os
import math
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from roi_detector import RoiDetector
from camelyon_dataset import MaskAnnotation
from resnet import resnet50

sys.path.append('./ASAP/bin/')
import multiresolutionimageinterface as mir

if __name__ == '__main__':
    tif_file = sys.argv[1]
    xml_file = sys.argv[2]
    mask_file = sys.argv[3]

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(tif_file)
    max_x, max_y = mr_image.getLevelDimensions(8)
    image = mr_image.getUCharPatch(0, 0, max_x, max_y, 8)
    
    anno = MaskAnnotation(xml_file)
    gt_mask = np.zeros((max_y,max_x), np.uint8)
    pts = []
    for attrib, contour in anno.contours:
        contour = contour/256
        contour = contour.astype(np.int32)
        pts.append(contour)
    cv2.fillPoly(gt_mask, pts, 255)

    predict = np.load(mask_file)
    predict = cv2.resize(predict*255, None, fx=1/8.0, fy=1/8.0)
    predict = predict.astype(np.uint8)

    cv2.imshow('groundtruth', gt_mask)
    cv2.imshow('image', image)
    cv2.imshow('predict', predict)
    cv2.waitKey()
