import numpy as np


class LeastSquaresLinearRegressor(object):

    def __init__(self):
        ''' Constructor of an sklearn-like regressor
        '''

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares 

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
            \min_{w,b}  \sum_{n=1}^N (y_n - w^Tx_n - b)^2
        '''
        N, F = x_NF.shape
        add1 = np.ones(N);
        myX=x_NF.copy();
        myX=np.insert(x_NF, F, values=add1, axis=1);
        #print(myX)
        invT=np.linalg.inv(np.dot(myX.T,myX))
        #print(invT)
        Y=np.mat(y_N)
        newY=np.dot(myX.T, Y.T)
        allW=np.dot(invT, newY)
        #print(allW)
        self.w_F=allW[0:F];
        self.w_F=np.asarray(self.w_F).reshape(-1)
        self.b=allW[F,0];
        #print(self.w_F, self.b)
        #print(self.w_F.size, " ", self.b)

    def predict(self, x_NF):
        ''' Make prediction given input features x

        Args
        ----
        x_NF : 2D array, (n_examples, n_features) (N,F)
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_N : 1D array, size N
            Each value is the predicted scalar for one example
        '''
        N, F=x_NF.shape
        addb=np.ones(N)*self.b
        #addb=addb.reshape(addb.size,1)
        #print(addb)
        ans=np.dot(x_NF, self.w_F.reshape(self.w_F.size,1))
        ans=ans.reshape(1,addb.size)
        return np.asarray(addb+ans).reshape(-1)

    def print_weights_in_sorted_order(
            self, feat_name_list=None, float_fmt_str='% 7.2f'):
        ''' Print learned coeficients side-by-side with provided feature names

        Args
        ----
        feat_name_list : list of str
            Each entry gives the name of the feature in that column of x_NF
        
        Post condition
        --------------
        Printed all feature names and coef values side by side (one per line)
        Should print all values in w_F first.
        Final line should be the bias value.
        '''
        N1=0;
        if (feat_name_list!=None):
            N1=feat_name_list.size()
        N2=self.w_F.size;
        if N2>0:
            for i in range(0,N2):
                if (i<N1):
                    print(feat_name_list[i], ': ', self.w_F[i]);
                else:
                    print(self.w_F[i]);
            print(self.b)

if __name__ == '__main__':
    ## Simple example use case
    # With toy dataset with N=100 examples
    # created via a known linear regression model plus small noise

    prng = np.random.RandomState()
    N = 10000

    w_F = np.asarray([1.1, -2.2, 3.3])
    x_NF = prng.randn(N, 3)
    y_N = np.dot(x_NF, w_F) + 0.03 * prng.randn(N)+4.4
    #print(np.mat(y_N).T)
    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)
    linear_regr.print_weights_in_sorted_order()
    yhat_N = linear_regr.predict(x_NF)
    #print(yhat_N)
    #print(np.array([1,2,3]))