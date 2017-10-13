#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-11 20:49 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""

"""

import random
import os
import numpy as np


raw_fog = np.random.binomial(1, 0.1, 584)
raw_notfog = np.random.binomial(1, 0.1, 574)

for i, item in enumerate(os.listdir("./train/fog")):
    if raw_fog[i] == 1:
        #os.rename(item, "./test/fog/")
        os.system("mv ./train/fog/" + item + " ./test/fog/")
        print (item, "moved")
for i, item in enumerate(os.listdir("./train/notfog")):
    if raw_notfog[i] == 1:
        #os.rename(item, "./test/notfog/")
        os.system("mv ./train/notfog/" + item + " ./test/notfog/")
        print (item, "moved")





