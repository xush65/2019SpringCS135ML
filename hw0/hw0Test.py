# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:02:05 2019

@author: xush4
"""

from hw0 import split_into_train_and_test, calc_k_nearest_neighbors
import numpy as np
#tests'''

x_LF = np.arange(1000).reshape((100, 10));

'''
print(allx);

np.random.shuffle(allx)
A=allx[0:2]
print (A);
(s,r)=allx.shape;
print(s);
'''

'''
A,B=split_into_train_and_test(x_LF, 0.98);
print('A=',A,'\n\n');
'''

'''
x_LF = np.eye(50)
'''



train_MF1, test_NF1 = split_into_train_and_test(x_LF, frac_test=0.3, 
                            random_state=42)
train_MF3, test_NF3 = split_into_train_and_test(x_LF, frac_test=0.3, 
                            random_state=None)
train_MF2, test_NF2 = split_into_train_and_test(x_LF, frac_test=0.3, 
                            random_state=42)


print('A=',train_MF1[0], '\n\n', 'B=', train_MF2[0], '\n\n', train_MF3[0]);

'''
train_MF= np.array([[1,0],[0,1],[-1,0],[0,-1]])
test_NF =np.array([[0, -0.9]])
print('A=',train_MF, '\n\n', 'B=', test_NF, '\n\n');
'''

##print(np.random.RandomState(0).rand(4))
##print(np.random.RandomState(10).rand(4))
D=calc_k_nearest_neighbors(train_MF1, test_NF1, 1)

##for i in range (0,10):
##    print(D[i])
##print(D)