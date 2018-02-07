import random
import sys
import os

import multiprocessing

import cv2
import torch
from torch.utils.data import Dataset

from camelyon_dataset import CamelyonDataset
from preprocess import ReColor, RandomJitter, RandomRotate, RandomFlip


class DatasetWrapper(Dataset):

    def __init__(self, backend, max_iter):
        self.backend = backend
        self.max_iter = max_iter
        self.trans_im = [ ReColor() ]
        #self.trans_all = [ RandomFlip(), RandomJitter() ]
        self.trans_all = [ ]

    def __getitem__(self, index):
        image, mask = self.backend.next()
        for trans in self.trans_im:
            image = trans(image)
        for trans in self.trans_all:
            image,mask = trans(image,mask)
        mask = cv2.resize(mask, None, fx=1/16.0, fy=1/16.0)
        mask = torch.Tensor(mask) > 128
        image = torch.Tensor(image)
        image = image.permute(2,0,1)/255.0
        return image, mask
    
    def __len__(self):
        return self.max_iter

class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch_data):
        image_list = []
        mask_list = []
        for image, mask in batch_data:
            image_list.append(image)
            mask_list.append(mask)
        return torch.stack(image_list), torch.stack(mask_list)

def get_split_list():
    stage_anno = './experiment/train_set.txt'
    xml_root = './dataset/train/label/'
    tif_root = './dataset/train/image/'
    train_tif = []
    train_xml = []
    test_tif = []
    test_xml = []
    with open(stage_anno) as fin:
        for line in fin:
            node, stage, rtype = line.strip().split('\t')
            tif_file = os.path.join(tif_root, node+'.tif')
            xml_file = None if rtype=='None' else os.path.join(xml_root, node+'.xml') 
            if rtype == 'test':
                test_xml.append(xml_file)
                test_tif.append(tif_file)
            else:
                train_xml.append(xml_file)
                train_tif.append(tif_file)
    return train_tif,train_xml,test_tif,test_xml


def get_dataset(is_train=True, is_test=True):
    cv2.setNumThreads(0)
    train_tif,train_xml,test_tif,test_xml = get_split_list()
    if is_train:
        train_dataset = CamelyonDataset(train_tif, train_xml, 800,
                fg_ratio=1.0/2.0)
        train_dataset = DatasetWrapper(train_dataset, 10000000)
    else:
        train_dataset = None
    if is_test:
        print(test_tif)
        test_dataset = CamelyonDataset(test_tif, test_xml, 800,
                fg_ratio=1.0/2.0)
        test_dataset = DatasetWrapper(test_dataset, 10000000)
    else:
        test_dataset = None
    return train_dataset, test_dataset




