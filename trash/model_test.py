#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-12 13:22 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Model Test.
"""
import torch
from torchvision import datasets, models

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

#model.avgpool = torch.nn.Conv2d(500, 2, kernel_size=(3, 3), stride=1, padding=1)
#model.fc = torch.nn.AvgPool2d(kernel_size=num_ftrs, stride=0)

print (model)





