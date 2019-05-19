'''
hw0.py
Author: TODO

Tufts COMP 135 Intro ML

'''

import numpy as np
def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
    
    '''if random_state is None:'''
    rng = np.random.RandomState(random_state)
    '''
    else :
        rng = np.random.RandomState(random_state)
    '''
    R=x_all_LF.shape[0];
    N=np.int(np.ceil(R*frac_test))
    temp=x_all_LF.copy();
    rng.shuffle(temp)
    test_set=temp[0:N];
    train_set=temp[N:R];
    ##print(train_set[0][0])
    return train_set, test_set

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    
    LN=data_NF.shape[0];  ##Number of examples for nb
    LQ=query_QF.shape[0]; ##Number of candidates to find nb
    LF=query_QF.shape[1]; ##Length of features
    dist=np.zeros((LQ,LN))
    for i in range(0, LQ):
        for j in range (0, LN):
            ##print(query_QF[i-1],'\n',data_NF[j-1],'\n\n')
            dist[i][j]= np.linalg.norm(query_QF[i]-data_NF[j]);
    ##print(dist);
    
    nb=np.zeros((LQ, K , LF));
    index=range(0,LN);
    for i in range(0, LQ):
        dist_i=np.zeros((2,LN));
        dist_i[0]=index;
        dist_i[1]=dist[i];
        dist_i=dist_i.transpose();
        ##print(dist_i);
        dist_i=dist_i[dist_i[:,1].argsort()];
        ##print(dist_i);
        for j in range(0,K):
            t=np.int(dist_i[j][0]);
            ##print(t);
            nb[i][j]=data_NF[t];
    return nb