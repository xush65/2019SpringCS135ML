# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:09:02 2019

@author: xush4
"""

from LRGradientDescent import LogisticRegressionGradientDescent as LRGD
from LRGradientDescentWithFeatureTransform import LRGDWithFeatureTransform as LRGDF
from show_images import show_images
import numpy as np
from scipy.special import logsumexp
from scipy.special import expit as sigm #sigmoid function
from numpy import genfromtxt
from matplotlib import pyplot as plt

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.linear_model
import sklearn.tree
import sklearn.metrics

from scipy.special import expit as sigm
from numpy.random import randint

def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    ''' Compute counts of four possible outcomes of a binary classifier for evaluation.
    
    Args
    ----
    ytrue_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats
        Each entry represents a predicted binary value (either 0 or 1).
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    TP : float
        Number of true positives
    TN : float
        Number of true negatives
    FP : float
        Number of false positives
    FN : float
        Number of false negatives
    '''
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    FP_id=[]
    FN_id=[]
    l=ytrue_N.size
    for i in range(0,l):
        if (yhat_N[i]==1):
            if (ytrue_N[i]==1):
                TP=TP+1.0
            else:
                FP=FP+1.0
                #FP_id.append(i)
        else:
            if (ytrue_N[i]==0):
                TN=TN+1.0
            else:
                FN=FN+1.0
                #FN_id.append(i)      
    return TP, TN, FP, FN #, FP_id, FN_id

def make_noise(x,y):
    N=int(x[0,:].size)
    #print(N)
    x_all=x;
    y_all=y;
    x_on=0;
    for j in range(9):
        x_j=x
        for i in range(y.size):
            for k in range(randint(1,10)):
                pos=randint(0,N)
                x_j[i, pos]=1-x[i,pos]
        x_all=np.concatenate((x_all, x_j), axis=0)
        y_all=np.concatenate((y_all, y), axis=0)
        #print(x_all.shape)
    return x_all, y_all

#x= genfromtxt('data_sneaker_vs_sandal/x_train.csv', delimiter=',')[1:]
##xbias_NG = lr.insert_final_col_of_all_ones(x_all)
#y= genfromtxt('data_sneaker_vs_sandal/y_train.csv', delimiter=',')[1:]
#x_n, y_n=make_noise(x,y)
## Reshuffle:
#Data=np.concatenate((x_n, np.matrix(y_n).T), axis=1)
#np.random.shuffle(Data)
#x_n=Data[:,:-1]
#y_n=np.asarray(Data[:,-1]).reshape(-1)
#va_rate=0.3
#x_va0=x[:int(np.ceil(va_rate*y.shape[0])),]
#y_va0=y[:int(np.ceil(va_rate*y.shape[0]))]
#x_te0=x[int(np.ceil(va_rate*y.shape[0])):,]
#y_te0=y[int(np.ceil(va_rate*y.shape[0])):]
#orig_lr1 = LRGDF(alpha=10.0, step_size=0.1)
#orig_lr1.fit(x_te0, y_te0)

print(np.array([1,1])/np.array([2,2]))
va_rate=0.3
x_va=x_n[:int(np.ceil(va_rate*y_n.shape[0])),]
print(max(np.asarray(x_va[randint(0,30000),:]).reshape(-1)))
y_va=y_n[:int(np.ceil(va_rate*y_n.shape[0]))]
x_te=x_n[int(np.ceil(va_rate*y_n.shape[0])):,]
y_te=y_n[int(np.ceil(va_rate*y_n.shape[0])):]

y_copy=y.copy()
print(y.size)
print(x[2, 500:700])
#print(min(np.asarray(np.amax(x,axis=1))))
#print(x_va[0].size)
#print(np.sum(np.multiply((x_va[:,1:-1]>0.),(((x_va[:,:-2]<=0.) + (x_va[:,2:]<=0.0)==1))),axis=1))
##print(y_va)
##print(True+True)

new_lr = LRGDF(alpha=10.0, step_size=0.1)
new_lr.fit(x_te, y_te)

#y_hat0=np.asarray(new_lr.predict_proba(x_va)[:,1]).reshape(-1)
#print(y_hat0)
#tp, tn, fp, fn=calc_TP_TN_FP_FN(y_va, y_hat0>=0.5)
#acc=(tp + tn) / float(tp + tn + fp + fn + 1e-10)