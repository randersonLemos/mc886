import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class MyLinearRegression:
    def __init__(self, n_features):
        self.n_thetas = n_features + 1
        self.theta_0 = np.random.uniform(low=-1, high=1, size=1) * 0
        self.thetas = np.asmatrix(np.random.uniform(low=-1, high=1, size=n_features)).T * 0


    def get_thetas(self):
        return np.append(self.theta_0, self.thetas)


    def h_theta(self, x):
        return self.theta_0 + x*self.thetas


    def gradient(self, X, y):
        X = np.asmatrix(X)
        y = np.asmatrix(y)
        m = len(y)
        ht = self.h_theta(X)
        grad_theta_0 = 1/m * (ht - y).sum()
        grad_thetas = np.zeros(len(self.thetas)) 
        for i in range(len(grad_thetas)):
            grad_thetas[i] = 1/m * (ht - y).T*X[:,i]
        return grad_theta_0, grad_thetas


    def theta_update(self, grad_theta_0, grad_thetas, learning_rate):
        self.theta_0 = self.theta_0 - learning_rate*grad_theta_0
        self.thetas  = self.thetas  - learning_rate*(np.asmatrix(grad_thetas).T)


    def cost(self, X, y):
        X = np.asmatrix(X)
        y = np.asmatrix(y)
        m = len(y) 
        ht = self.h_theta(X)        
        return float( 1/2*m * (ht - y).T * (ht - y) )


    def fit(self, X, y, learning_rate, n_epochs, size_batches=-1, verbose=False):
        for epoch in range(n_epochs):
            n_batches = 1
            if size_batches != -1:
                n_batches = int( len(X) / size_batches )

            for en, (batchX, batchy) in enumerate( zip(np.array_split(X, n_batches), np.array_split(y, n_batches)) ):                
                grad_theta_0, grad_thetas = self.gradient(batchX, batchy)
                self.theta_update(grad_theta_0, grad_thetas, learning_rate=learning_rate)                

            if verbose and (epoch%10 == 0):
                print('Epoch: {:04d} | Num. batches: {} | Cost function: {}'.format(epoch, n_batches, self.cost(X, y)))


if __name__ == '__main__':
    #X = pd.read_csv('X.csv')
    #y = pd.read_csv('y.csv'); y = y[['porc_ACERT_lp']]

    #X = (X - X.min()) / (X.max() - X.min())

    from sklearn.linear_model import LinearRegression


    X = np.asmatrix( np.array([ [1], [2], [3], [4] ]) )
    y = np.asmatrix( np.array([ [0.5], [1]  , [1.5], [2.0] ]) )

    reg = LinearRegression(fit_intercept=True).fit(X,y)


    lr = MyLinearRegression(n_features=X.shape[1])
    lr.fit(X, y, learning_rate=0.001, n_epochs=100000, verbose=True)

    print( reg.coef_ )
    print( np.array(np.linalg.inv(X.T*X)*X.T*y).squeeze() )
    print( lr.get_thetas() )


    import IPython; IPython.embed()
