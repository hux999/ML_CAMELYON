import sys
import os

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from evaluator import EvalPrecision,EvalRecall
from data import CollateFn, get_dataset
from resnet import resnet50,resnet101



def Test(net, dataset):
    net.eval()
    net.cuda()
    while True:
        print('forward')
        image, mask = dataset[0] # next
        org_image = image.permute(1,2,0).numpy()*255
        img_h, img_w, _ = org_image.shape
        image = Variable(image.unsqueeze(0)).cuda()
        predict = F.softmax(net(image), dim=1)[0, 1, :, :].squeeze()*255
        predict = predict.cpu().data.numpy().astype(np.uint8)
        cv2.imshow('predict', cv2.resize(predict, (img_w,img_h)))
        cv2.imshow('image', org_image.astype(np.uint8) )
        cv2.imshow('mask', cv2.resize(mask.numpy().astype(np.uint8)*255, (img_w, img_h)))
        cv2.waitKey()

if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset(is_train=False, is_test=True)
    net = resnet101(num_classes=2, pretrained=False)
    net.load_state_dict(torch.load('model/epoch_40000.pt'))
    Test(net, test_dataset)

