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
from resnet import resnet50,resnet101

sys.path.append('./ASAP/bin/')
import multiresolutionimageinterface as mir

class WSITraveler(Dataset):

    def __init__(self, tif_file, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride
        self.tif_file = tif_file
        self.load_image()
        self.all_patch = self.get_all_patch(tif_file)

    def get_all_patch(self, tif_file):
        all_patch = []
        patch_size = self.patch_size
        stride = self.stride
        roi = RoiDetector(tif_file)
        max_x, max_y = self.max_size
        for x in range(0, max_x-stride, stride):
            for y in range(0, max_y-stride, stride):
                cx = x + patch_size/2
                cy = y + patch_size/2
                if roi.is_foreground(cx,cy):
                    all_patch.append([x, y])
        return all_patch

    def is_empty(self, image_patch):
        check = image_patch == image_patch[0,0,:]
        return check.all()


    def load_image(self):
        reader = mir.MultiResolutionImageReader()
        self.mr_image = reader.open(self.tif_file)
        self.max_size = self.mr_image.getDimensions()

    def __len__(self):
        return len(self.all_patch)

    def __getitem__(self, index):
        x, y = self.all_patch[index]
        try_count = 0
        while True:
            image_patch = self.mr_image.getUCharPatch(x, y,
                    self.patch_size, self.patch_size, 0)
            if not self.is_empty(image_patch) or try_count>10:
                break
            print('empty_patch')
            try_count += 1
            self.load_image()
        image_patch = torch.Tensor(image_patch)
        image_patch = image_patch.permute(2,0,1)/255.0
        return (x,y), image_patch


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch_data):
        pts_list = []
        image_list = []
        for pts, image in batch_data:
            pts_list.append(pts)
            image_list.append(image)
        return pts_list, torch.stack(image_list)

def PredictWSI(net, tif_file):
    print(tif_file.split('/')[-1][:-4])
    # setup model
    net.eval()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    # setup data
    patch_size = 1024
    stride = 512
    dataset = WSITraveler(tif_file, patch_size, stride)
    batch_size = 16
    num_data = len(dataset)
    mask_size = dataset.max_size
    mask_w = int(math.ceil(1.0*mask_size[0]/stride)*stride/32)
    mask_h = int(math.ceil(1.0*mask_size[1]/stride)*stride/32)
    mask = np.zeros((mask_h, mask_w), np.float32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=batch_size, collate_fn=CollateFn())
    start = time.time()
    for i_batch, (pts_list, image_data) in enumerate(dataloader):
        image_data = Variable(image_data.cuda())
        predict = F.softmax(net(image_data), dim=1)[:, 1, :, :] # /8
        predict = F.avg_pool2d(predict, kernel_size=2, stride=2) # /32
        predict = predict.squeeze()
        predict = predict.cpu().data.numpy()
        for i, (x, y) in enumerate(pts_list):
            x = x/32
            y = y/32
            s = patch_size/32
            mask[y:y+s, x:x+s] = np.maximum(mask[y:y+s, x:x+s], predict[i, :, :])
            #mask[y:y+s, x:x+s] = predict[i, :, :]
        end = time.time()
        print('%d/%d %.3f per patch' % (i_batch*batch_size, num_data, 
            (end-start)/(i_batch*batch_size+batch_size)))
        if i_batch % 10 == 0:
            smask = cv2.resize(mask*255, None, fx=1/8.0, fy=1/8.0)
            smask = smask.astype(np.uint8)
            cv2.imshow('mask', smask)
            cv2.waitKey(100)
    np.save(tif_file.split('/')[-1][:-4], mask)


if __name__ == '__main__':
    net = resnet101(num_classes=2, pretrained=False)
    net.load_state_dict(torch.load('model/epoch_30000.pt'))
    PredictWSI(net, sys.argv[1]) 
