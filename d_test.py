#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-11 20:25 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""

"""


from torchvision import datasets

im_datasets = {"train": datasets.ImageFolder("train")}

for im in im_datasets["train"]:
    print (im)

