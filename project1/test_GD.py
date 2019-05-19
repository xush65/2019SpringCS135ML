from LRGradientDescent import LogisticRegressionGradientDescent as LRGD
import numpy as np
from scipy.special import logsumexp
from scipy.special import expit as sigm
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

a=np.matrix([[1,-1],[2,-2]])
a=np.vstack((a,np.array([2,2])))
print(a)
#print(np.column_stack([0,np.array([1,-1])]))
lr=  LRGD()
arr=np.matrix('0.1 0.2 0.1 0.4;0.33 0.4 0.5 0.2;0.33 0.4 0.5 0.2')
#print(np.log(arr))
arr2=np.matrix('1 0; 3 8').T
w=np.matrix([1,2,3,4,5])
w2=np.array([0,1,2,3])
b=2;
#print(sigm(-10000))
#print(np.dot(w2,w)[0,0])
x=np.dot(arr,w2)+b
#print(x)

a=sigm(x)
#print(a, w2)
#print(w2.dot(1-a))
#print(1-a)

w_g=np.array([0,1])
w_G=np.array([(1e12)*np.random.rand(),(1e12)*np.random.rand()])
#x=np.matrix([[0,1,2,-3,1],[0,1,-2,3,1],[0,-3,2,3,1],[0,-1,2,-3,1],[2.05,-1,2,3,1],[0,1,2,-3,1],[-8.21,1,2,3,1]])
#y_N=np.array([0,0,0,0,0,1,1])
#print(x.T)
x=np.column_stack([np.random.random(300), np.ones(300)])
#print(x)
y_N=np.random.choice([1,0], 300)
print(lr.calc_loss(w_G, x, y_N))
print(lr.calc_grad(w_G, x, y_N))

z=1.0*np.array([-19998, -19998, -19998, -19998, -19998, -19998, -19998, -19998, -19998,
 -19998.])
print(z.max())
Nint=len(z)
zeroN= np.zeros(Nint)
print(np.logaddexp.reduce(arr))

#
#x_NF = np.hstack([np.linspace(-2, -1, 5), np.linspace(1,2, 5)])[:,np.newaxis]
#y_N = np.hstack([np.zeros(5), 1.0 * np.ones(5)])
#lr=LogisticRegression(C=10)
#lr.fit(x_NF,y_N)
#print(np.hstack((lr.coef_,lr.intercept_[:,None])))
#[[ 2.78299613e+00 -2.35092232e-17]]