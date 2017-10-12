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

MODEL_NAME="./fixmodel_best.pth.tar"
#MODEL_NAME="./fixmodel_best.pth.tar"

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
im_dataset = {"test": datasets.ImageFolder(os.path.join(data_dir, "test"), data_transform["test"])}
dataloader = {"test": torch.utils.data.DataLoader(im_dataset["test"], batch_size=8, shuffle=True, num_workers=4)}

dataset_sizes = {"test": len(im_dataset["test"])}

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_gpu = torch.cuda.is_available()
use_gpu = False

def eval(model, criterion):
    since = time.time()
    running_loss = 0.0
    running_corrects = 0.0

    for data in dataloader["test"]:
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        print (" | Loss: ", loss.data[0])

        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    final_loss = running_loss / dataset_sizes["test"]
    final_acc = running_corrects / dataset_sizes["test"]
    print (" | Final loss: {}, Final accuracy: {}".format(final_loss, final_acc))
    print (" | Consume time {}s".format(time.time() - since))
    print (" | Done!")


model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(model_weights)
if use_gpu:
    model = model.cuda()

eval(model, nn.CrossEntropyLoss())

