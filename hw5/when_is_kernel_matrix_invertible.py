'''
This script will create N distinct feature vectors,
then compute the N x N kernel matrix "K" for this dataset,
then determine if the kernel matrix "K" is invertible.

Usage
-----
Call as a script with no arguments

>>> python when_is_kernel_matrix_invertible.py

'''

import numpy as np

def k_rbf(xa, xb, gamma=1.0):
    return np.exp(-gamma * np.sum(np.square(xa-xb)))

def k_linear(xa, xb):
    return np.sum(xa * xb)


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=0)
    N = 4
    F = 1

    ## Create dataset of N distinct points
    if F == 1:
        x_NF = np.asarray([-3, -1, 2, 4])[:,np.newaxis]
    else:
        x_NF = np.random.randn(N, F)

    N = x_NF.shape[0]
    print("Using N=%d distinct data vectors" % N)
    for n in range(N):
        print("x[%d] = %s" % (n, x_NF[n]))

    ## Create kernel matrix for training set, with shape (N, N)
    K_linear_NN = np.zeros((N, N))
    K_rbf_NN = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            K_linear_NN[a, b] = k_linear(x_NF[a], x_NF[b])
            K_rbf_NN[a, b] = k_rbf(x_NF[a], x_NF[b])

    for (name, K_NN) in [('linear', K_linear_NN), ('rbf', K_rbf_NN)]:
        print("\nKernel: %s" % name)
        print(K_NN)
        print("is it invertible? if so, determinant must be NOT equal to 0")
        det = np.linalg.det(K_NN)
        print("det = %.5f" % det)
        print("so this Kernel matrix is %s" % (
            "NOT invertible" if np.allclose(det, 0.0) else "YES INVERTIBLE"))



