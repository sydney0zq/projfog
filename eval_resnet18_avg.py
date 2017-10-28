#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-12 09:56 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Evalution for fog project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms, models
import torchvision
import time
import os
from PIL import Image, ImageFile

MODEL_NAME="./model_avgpool_best.pth.tar"

if os.path.exists(MODEL_NAME):
    model_weights = torch.load(MODEL_NAME)
else:
    raise IOError

data_transform = {
    "test": transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),        
}

data_dir = '.'
dataloader = {"test": torch.utils.data.DataLoader(im_dataset["test"], batch_size=1, shuffle=False, num_workers=1)}

dataset_sizes = {"test": len(im_dataset["test"])}

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_gpu = torch.cuda.is_available()
#use_gpu = False

def eval(im_path, model, criterion):
    since = time.time()
    running_loss = 0.0
    running_corrects = 0.0
    diff = []

    im = Image.open(im_path)
    label = torch.Tensor((0, 1))

    for ii, data in enumerate(dataloader["test"]):
        (inputs, labels) = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
#        print (" | Loss: ", loss.data[0])
        #labels_ = labels.type(torch.cuda.LongTensor)

        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

        diff = np.array(preds) - np.array(labels.data)
        print (diff)
        for i in range(len(diff)):
            if i != 0:
                im = np.array(inputs[i].data) * 255
                im = im.astype(np.int8)
                Image.fromarray(im).save(str(ii) + str(i) + ".jpg")

    final_loss = running_loss / dataset_sizes["test"]
    final_acc = running_corrects / dataset_sizes["test"]
    #print (" | Final loss: {}, Final accuracy: {}".format(final_loss, final_acc))
    print (" | Accuracy: {}".format(final_acc))
    print (" |      Consume time {}s".format(time.time() - since))
    return final_acc


model = models.resnet18(pretrained=True)

class novelmodel(nn.Module):
    def __init__(self):
        super(novelmodel, self).__init__()
        self.features = nn.Sequential(
            *list(model.children())[:-2]
        )
        self.conv1 = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=2)
        self.avgpool = torch.nn.AvgPool2d(4)
    def forward(self, x):
        #print ("Feature size: {}".format(x.size()))
        x = self.features(x)
        x = self.conv1(x)
        #print ("Conv1 size: {}".format(x.size()))
        x = self.avgpool(x)
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x[:, :, 0, 0]
        return x

model = novelmodel()


model.load_state_dict(model_weights)
if use_gpu:
    model = model.cuda()

acc = 0
evaltimes = 10
for i in range(evaltimes):
    acc += eval(model, nn.CrossEntropyLoss())
print (" | Final accuracy: {}%".format(acc/evaltimes*100))

