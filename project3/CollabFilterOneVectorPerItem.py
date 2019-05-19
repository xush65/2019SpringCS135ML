# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from utils import load_dataset

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        random_state = self.random_state # inherited
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users),
            c_per_item=ag_np.ones(n_items),
            U=random_state.randn(n_users,self.n_factors),
            V=random_state.randn(n_items,self.n_factors),
            )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        yhat_N = ag_np.ones(N)*mu+b_per_user[user_id_N]+c_per_item[item_id_N]\
                +ag_np.einsum('ij, ij->i', U[user_id_N], V[item_id_N])
        
        return yhat_N

    def penalty(self,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ###print(V.shape, U.shape, b_per_user.shape, c_per_item.shape)
        U_vec=ag_np.einsum('ij, ij->i', U,U)
        V_vec=ag_np.einsum('ij, ij->i', V,V)
        my_sum=0.0\
            +ag_np.dot(V_vec,V_vec)\
            +ag_np.dot(U_vec,U_vec)\
            +ag_np.dot(b_per_user,b_per_user)\
            +ag_np.dot(c_per_item,c_per_item);
        return my_sum
    
    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        
        loss_total = ag_np.dot(y_N-yhat_N, y_N-yhat_N)+self.alpha*\
            self.penalty(**param_dict)
        return loss_total    

if __name__ == '__main__':
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_dataset()
    model = CollabFilterOneVectorPerItem(
        n_factors=2, alpha=1,
        n_epochs=250, step_size=0.25)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)

#### 3b
#K=[0,2,10,50]
#a=np.logspace(-10,-5)
#tr_error=[];
#va_error=[];
#for i in range(4):
#    te=[]; ve=[];
#    for j in range(int(a.size)):
#        train_tuple, valid_tuple, test_tuple, n_users, n_items = load_dataset()
#        model3b = CFV(n_epochs=50, step_size=0.5, n_factors=K[i], alpha=a[j])
#        model3b.init_parameter_dict(n_users, n_items, train_tuple)
#        model3b.fit(train_tuple, valid_tuple)
#        te.append(model3b.trace_mae_train[-1])
#        ve.append(model3b.trace_mae_valid[-1])
#    tr_error.append(te)
#    va_error.append(ve)