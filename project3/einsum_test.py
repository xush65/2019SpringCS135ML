# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:29:24 2019

@author: xush4
"""

import autograd.numpy as ag_np
import numpy as np
a = np.arange(12).reshape(4,3)
b = np.arange(5)
c = np.arange(12).reshape(4,3)

d=np.einsum('ij, ij->i', a, c)

