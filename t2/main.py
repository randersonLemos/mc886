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


class LinearRegression:
    def __init__(self, n_thetas, n_degrees):
        self.n_thetas = n_thetas
        self.n_degrees = n_degrees

        #self.theta_0 = np.random.uniform(low=-1.0, high=1.0, size=1)
        #self.thetas = np.asmatrix(
        #    np.random.uniform(low=-1.0, high=1.0, size=n_thetas)
        #).T

        self.theta_0 = 1
        self.thetas = np.asmatrix([[1],[1],[1],[1]])


    def h_theta(self, x):
        return self.theta_0 + x*self.thetas


    def h_theta_gradient(self, X, y):
        X = np.asmatrix(X.values)
        y = np.asmatrix(y.values).T
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
        y = np.asmatrix(y.values).T
        m = len(y) 

        ht = self.h_theta(X)
        
        return float( 1/2*m * (ht - y).T * (ht - y) )


if __name__ == '__main__':
    X = pd.read_csv('X.csv'); y = pd.read_csv('y.csv')

    X = X[['Q1', 'Q2', 'Q3', 'Q4']].iloc[:5,:]
    y = y[['nivel_profic_lp']].iloc[:5, 0]
    lr = LinearRegression(4, 1)

    cost = lr.cost(X, y)
    grad_theta_0, grad_thetas = lr.h_theta_gradient(X, y)
    lr.h_theta_update(grad_theta_0, grad_thetas, learning_rate=0.1)

    for i in range(100):
        print(lr.cost(X,y))
        print(lr.theta_0, lr.thetas)

        grad_theta_0, grad_thetas = lr.h_theta_gradient(X, y)
        lr.h_theta_update(grad_theta_0, grad_thetas, learning_rate=0.01)

        print('###')

        
    import IPython; IPython.embed()
