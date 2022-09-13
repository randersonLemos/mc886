import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocessing():
    data_path = "assets/SARESP_train.csv"
    df=pd.read_csv(data_path)

    for col in df.columns:
        na_percentage = df[col].isna().sum() / df.shape[0] * 100
        if na_percentage > 0:
            print(f'Column Name: {col}, Null Values Percentage: {na_percentage}')

    df = df.drop(['NEC_ESP_1', 'NEC_ESP_2', 'NEC_ESP_3', 'NEC_ESP_4', 'NEC_ESP_5', 'CD_ALUNO'], axis=1)

    le = LabelEncoder()
    columns_names = df.columns.to_list()
    column_object_names = []
    for column_name in columns_names:
        if df[str(column_name)].dtype == object:
            df[str(column_name)] = le.fit_transform(df[str(column_name)])


    # df.reset_index(inplace=True)
    X = df.drop(['nivel_profic_lp', 'nivel_profic_mat','nivel_profic_cie'],axis=1)
    y = df[['nivel_profic_lp', 'nivel_profic_mat', 'nivel_profic_cie']]

    X.to_csv('X.csv'); y.to_csv('y.csv')


class MyLinearRegression:
    def __init__(self, n_thetas):
        self.n_thetas = n_thetas

        self.theta_0 = np.random.uniform(low=-1, high=1, size=1)
        self.thetas =  np.asmatrix(np.random.uniform(low=-1, high=1, size=n_thetas)).T


    def h_theta(self, x):
        return self.theta_0 + x*self.thetas


    def h_theta_gradient(self, X, y):
        X = np.asmatrix(X.values)
        y = np.asmatrix(y.values)
        m = len(y)
        ht = self.h_theta(X)

        grad_theta_0 = 1/m * (ht - y).sum()

        grad_thetas = np.zeros(len(self.thetas)) 
        for i in range(len(grad_thetas)):
            grad_thetas[i] = 1/m * (ht - y).T*X[:,i]

        return grad_theta_0, grad_thetas


    def h_theta_update(self, grad_theta_0, grad_thetas, learning_rate):
        self.theta_0 = self.theta_0 - learning_rate*grad_theta_0

        self.thetas = self.thetas - learning_rate*(np.asmatrix(grad_thetas).T)


    def cost(self, X, y):
        X = np.asmatrix(X.values)
        y = np.asmatrix(y.values)
        m = len(y) 

        ht = self.h_theta(X)
        
        return float( 1/2*m * (ht - y).T * (ht - y) )


    def fit(self, X, y, learning_rate, n_epochs, size_batches=10000, verbose=False):
        for epoch in range(n_epochs):
            n_batches = int( len(X) / size_batches )

            grad_theta_0_lst = []
            grad_thetas_lst = []
            for en, (batchX, batchy) in enumerate( zip(np.array_split(X, n_batches), np.array_split(y, n_batches)) ):
                
                grad_theta_0, grad_thetas = self.h_theta_gradient(batchX, batchy)

                grad_theta_0_lst.append(grad_theta_0)
                grad_thetas_lst.append(grad_thetas)

                self.h_theta_update(grad_theta_0, grad_thetas, learning_rate=learning_rate)
                
                if verbose and (epoch%100 == 0):
                    print('Processing batch {}'.format(en+1))
                    print('Cost function at step {:06d}: {:012.6f}'.format(epoch, self.cost(batchX, batchy)))


if __name__ == '__main__':
    X = pd.read_csv('X.csv'); y = pd.read_csv('y.csv')

    #X = X[['Q1', 'Q2', 'Q3', 'Q4']]
    y = y[['porc_ACERT_lp']]
    lr = MyLinearRegression(n_thetas=X.shape[1])
    lr.fit(X, y, learning_rate=0.000000000001, n_epochs=10000, verbose=True)
