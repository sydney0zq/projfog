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
IM_DIR="./test/fog"
#IM_DIR="./test/notfog"


if os.path.exists(MODEL_NAME):
    model_weights = torch.load(MODEL_NAME)
else:
    raise IOError

data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '.'

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_gpu = torch.cuda.is_available()
use_gpu = False

def eval(im_path, model):
    model.eval()
    since = time.time()

    inputs = data_transform(Image.open(im_path))
    inputs = inputs.unsqueeze(0)
    # Fog: 0
    # NotFog: 1

    if use_gpu:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    #print (" |      Consume time {}s".format(time.time() - since))
    return preds.data[0]

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

for im_name in os.listdir(IM_DIR):
    if im_name.endswith("jpg") or im_name.endswith("jpeg") or im_name.endswith("png"):
        pred = eval(os.path.join(IM_DIR, im_name), model)
        if pred == 0:
            print (" | {} is foggy.".format(im_name))
        else:
            print (" | {} is not foggy.".format(im_name))

