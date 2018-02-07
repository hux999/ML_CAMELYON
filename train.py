import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from evaluator import EvalPrecision,EvalRecall
from data import CollateFn, get_dataset
from resnet import resnet50,resnet101
from FocalLoss import FocalLoss

#from FocalLoss import FocalLoss


def SegLoss(predict, label):
    #loss = F.binary_cross_entropy_with_logits(predict.squeeze(), label)
    predict = predict.permute(0, 2, 3, 1).contiguous()
    predict = predict.view(-1, 2)
    label = label.view(-1)
    #loss = FocalLoss(2)(predict, label.long())
    loss = F.cross_entropy(predict, label.long())
    return loss


def Train(train_data, val_data, net, max_iter, lr, use_cuda=True):
    if use_cuda is not None:
        net.cuda()
    net_ = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    #net_ = net
    net_.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    dataloader = DataLoader(train_data, batch_size=20, shuffle=False, num_workers=12,
            collate_fn=CollateFn())
    dataloader = iter(dataloader)

    for i_iter in range(max_iter):
        # train
        image, mask = dataloader.next()
        image = Variable(image)
        mask = Variable(mask)
        if use_cuda:
            image = image.cuda()
            mask = mask.cuda()
        # forward
        predict = net_(image)
        loss = SegLoss(predict, mask)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_iter % 20 == 0:
            print(('iter:%d, loss:%f')  % (i_iter, loss.data[0]))

        # save model for each epoch
        if i_iter % 5000 == 0:
            torch.save(net.state_dict(), ('model/epoch_%d.pt' % i_iter))


if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset(is_train=True, is_test=False)
    net = resnet101(num_classes=2, pretrained=True)
    #net = resnet101(num_classes=2, pretrained=False)
    #net.load_state_dict(torch.load('./model/epoch_15000.pt'))

    Train(train_dataset, test_dataset, net,
        max_iter=100000, lr=0.0001, use_cuda=True)

