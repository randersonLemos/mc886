from typing import List
from abc import ABC,abstractmethod
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


class BaseFunction(ABC):
    @abstractmethod
    def __call__(self, X):
        pass
        
    @abstractmethod
    def grad(self, X):
        pass


class ReLU(BaseFunction):    
    def __call__(self, X):
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X):
        return np.where(X >= 0, 1, 0)


class LeakyRelu(BaseFunction):    
    def __call__(self, X, alpha=0.01):
        return np.maximum(alpha * X, X)
        
    def grad(self, X, alpha=0.01):
        return np.where(x>0, 1, alpha)
        

class Softmax(BaseFunction):    
    def __call__(self, X):
        """
        Arguments:
        X: (np.array) input data

        Return:
        Softmax output
        """    
        ##################################
        # TODO: implement here the Softmax
        ##################################
        max_ = np.max(X, axis=1, keepdims=True)
        exp = np.exp(X - max_)
        sum_ = np.sum(exp, axis=1, keepdims=True)
        return exp / sum_ 
        
    def grad(self, X):
        return 1 # discard this gradient


class CrossEntropy(BaseFunction):    
    def __call__(self, Y, Y_pred):
        """
        Arguments:
        Y: (np.array) ground-truth labels
        Y_pred: (np.array) predicted labels

        Return:
        Cross-Entropy output
        """ 
        ##################################
        # TODO: implement here the Cross-Entropy
        ##################################
        epsilon = 1e-8
        pred = np.clip(Y_pred, epsilon, 1. - epsilon)
        cross_entropy = -np.sum(Y * np.log(pred+epsilon)) / pred.shape[0]
        return cross_entropy

    def grad(self, Y, Y_pred):
        return Y_pred - Y # gradient with respect to Softmax's input


class Model:
    def __init__(self, layers_dims: List[int], 
                 activation_funcs: List[BaseFunction],
                 initialization_method: str = "random"):
        """
        Arguments:
        layers_dims: (list) a list with the size of each layer
        activation_funcs: (list) a list with the activation functions 
        initialization_method: (str) indicates how to initialize the parameters

        Example:
        
        # a model architecture with layers 2 x 1 x 2 and 2 ReLU as activation functions
        >>> m = Model([2, 1, 2], [ReLU(), ReLU()])
        """

        assert all([isinstance(d, int) for d in layers_dims]), \
        "It is expected a list of int to the param ``layers_dims"

        assert all([isinstance(a, BaseFunction) for a in activation_funcs]), \
        "It is expected a list of BaseFunction to the param ``activation_funcs´´"
        
        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self.initialize_model(initialization_method)


    def __len__(self):
        return len(self.weights)


    def initialize_model(self, method="random"):
        """
        Arguments:
        layers_dims: (list) a list with the size of each layer
        method: (str) indicates how to initialize the parameters

        Return: a list of matrices (np.array) of weights and a list of 
        matrices (np.array) of biases.
        """
        
        weights = []
        bias = []
        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
            
            # the weight w_i,j  connects the i-th neuron in the current layer to
            # the j-th neuron in the next layer          
            W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
            b = np.random.randn(1, self.layers_dims[l + 1])
            
            # He et al. Normal initialization
            if method.lower() == 'he':
                W = W * np.sqrt(2/self.layers_dims[l])
                b = b * np.sqrt(2/self.layers_dims[l])

            ###################################################
            # TODO: implement another initialization method
            #   ...
            ###################################################
            if method.lower() == 'xavier':
                # The Xavier inicialization is equals to he, but considers the n of inputs and outputs.
                #Whiles the He inicialization only considers the inputs
                W = W * np.sqrt(2/self.layers_dims[l + 8])
                b = b * np.sqrt(2/self.layers_dims[l + 8])

            weights.append(W)
            bias.append(b)

        return weights, bias


    def forward(self, X):
        """
        Arguments:
        X: (np.array) input data

        Return:
        Predictions for the input data (np.array)
        """      
        activation = X
        self.activations = [X]
        self.Z_list = []

        for weights, bias, activation_func in zip(self.weights, self.bias, self.activation_funcs):
            Z = np.dot(X, weights) + bias
            A = activation_func(Z)
            X = A

            self.Z_list.append(Z)
            self.activations.append(A)
        return X


class BaseOptimizer(ABC):    
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def step(self, grads):
        """
        Arguments:
        grads: (list)  a list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.array format.
        
        Return: 
        """
        pass


class SGDOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self, grads: List):
        """
        Arguments:
        grads: (list)  a list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.array format.
        
        Return: 
        """
        for i, (grad_w, grad_b) in enumerate(grads):
            self.model.weights[i] = self.model.weights[i] - self.lr * grad_w
            self.model.bias[i] = self.model.bias[i] - self.lr * grad_b
    

class AdamOptimizer(BaseOptimizer):
    pass
    

class Trainer:
    def __init__(self, model: Model, optimizer, loss_func: CrossEntropy):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = 0

    def backward(self, Y):
        """
        Arguments:
        Y: (np.array) ground truth/label vector.

        Return: 
        A list of tuples of matrices (weights' gradient, biases' gradient) both in np.array format.
        The order of this list should be the same as the model's weights. 
        For example: [(dW0, db0), (dW1, db1), ... ].
        """
        ############################################################
        # TODO: implement here the backward step.
        ############################################################
        m   = Y.shape[0]

        L    = len(self.model.activations) - 1
        WL   = None
        aL   = self.model.activations[L]
        aLm  = self.model.activations[L-1]
        zLm  = self.model.Z_list[L-1]
        gLm  = self.model.activation_funcs[L-1]
        dgLm = gLm.grad(zLm)

        delta = (aL - Y)*dgLm
        dw = np.dot(aLm.T, delta) / m
        db = np.sum(delta, axis=0) / m
        grads = [ (dw, db, ) ]

        #print('+++')
        #print('-dw-')
        #print(dw)
        #print('-db-')
        #print(db)

        while L > 1:
            L = L - 1
            WL    = self.model.weights[L]
            aLm   = self.model.activations[L-1]
            zLm   = self.model.Z_list[L-1]
            gLm   = self.model.activation_funcs[L-1]
            dgLm  = gLm.grad(zLm)

            delta = np.dot(delta, WL.T)*dgLm
            dw  = np.dot(aLm.T, delta) / m
            db = np.sum(delta, axis=0) / m
            grads.append( (dw, db, ) )

        #for dw, db in grads:
        #    print('-dw-')
        #    print(dw)
        #    print('-db-')
        #    print(db)

    def backward_(self, Y):
        y_pred = self.model.activations[-1]
        diff = self.loss_func.grad(Y, y_pred)
        weights_and_bias_grads = []
        
        for i, weight in enumerate(reversed(self.model.weights)):
            dw = np.dot(self.model.activations[-i -2].T, diff) * (1 / self.batch_size) 
            print('-dw-')
            print(dw)
            db = np.sum(diff, axis=0) * (1 / self.batch_size)
            print('-db-')
            print(db)
            #weights_and_bias_grads.append((dw,db))
            weights_and_bias_grads.append(dw)
            
            if i < len(self.model.weights) -1 :
                diff = np.dot(diff, weight.T) * self.model.activation_funcs[-i -2].grad(self.model.Z_list[-i -2])
                
        return weights_and_bias_grads[::-1]


    def train(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """
        Arguments:
        n_epochs: (int) number of epochs
        train_loader: (DataLoader) train DataLoader
        val_loader: (DataLoader) validation DataLoader

        Return: 
        A dictionary with the log of train and validation loss along the epochs
        """
        log_dict = {'epoch': [], 
                   'train_loss': [], 
                   'val_loss': [],
                   'train_acc':[],
                   'val_acc': []}

        self.batch_size = train_loader.batch_size
        for epoch in tqdm(range(n_epochs)):
            train_loss_history = []
            train_gts = []
            train_preds = []
            for i, batch in enumerate(train_loader):                
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()                         
                Y_pred = self.model.forward(X)

                train_loss = self.loss_func(Y, Y_pred)
                train_gts.append(Y)
                train_preds.append(Y_pred)
                train_loss_history.append(train_loss)

                grads = self.backward(Y)
                self.optimizer.step(grads)

            val_loss_history = []
            val_gts = []
            val_preds = []
            for i, batch in enumerate(val_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)
                val_loss = self.loss_func(Y, Y_pred)
                val_loss_history.append(val_loss)
                
                val_gts.append(Y)
                val_preds.append(Y_pred)         

            # appending losses to history
            train_loss = np.array(train_loss_history).mean()
            val_loss = np.array(val_loss_history).mean()
            
            train_gts = np.concatenate(train_gts, axis=0)
            train_preds = np.concatenate(train_preds, axis=0)
            
            val_gts = np.concatenate(val_gts, axis=0)
            val_preds = np.concatenate(val_preds, axis=0)
            
            train_acc = balanced_accuracy_score(train_gts, train_preds)
            val_acc = balanced_accuracy_score(val_gts, val_preds)
            
            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)
            log_dict['train_acc'].append(train_acc)
            log_dict['val_acc'].append(val_acc)
        
        return log_dict


if __name__ == '__main__':
    ######## checking Backward pass ########
    
    # architecture: 2 x 1 x 2
    m = Model([2, 1, 2], [ReLU(),Softmax()])
    
    X = np.array([
          [0 ,1]
        , [-1,0]
        #, [0.5,0.5]
    ])

    y = np.array([
          [0,1]
        , [1,0]
        #, [1,1]
    ])
 
    
    W0 = np.array([[2],
                   [1]])
    b0 = np.array([[1]])
    W1 = np.array([[2, 3]])
    b1 = np.array([[1, -1]])
    
    m.weights = [W0, W1]
    m.bias = [b0, b1]
    
    t = Trainer(m, None, CrossEntropy())
    t.batch_size = X.shape[0]
    
    prediction = m.forward(X)
    grads_ = t.backward_(y)
    grads = t.backward(y)
    #import pprint
    #pprint.pprint(grads)
    #pprint.pprint(grads_)
    #import IPython; IPython.embed()
    
    # We let this value just in case you need to check your results
    #print(updated_weights_bias)
    # expected_dZ1 = np.array([[ 0.5       , -0.5       ],
    #                         [-0.11920292,  0.11920292]])
    #
    # expected_dZ0 = np.array([[-0.5],
    #                          [ 0. ]])
    #
    # y_pred = np.array([[0.5       , 0.5       ],
    #                    [0.88079708, 0.11920292]])
    
    
    #expected_dW1 = np.array([[ 0.5, -0.5]])
    #
    #expected_db1 = np.array([[ 0.19039854, -0.19039854]])
    #
    #expected_dW0 = np.array([[ 0.  ],
    #                         [-0.25]])
    #
    #expected_db0 = np.array([[-0.25]])
    #
    #dW1, db1 = grads[1]
    #assert (abs(expected_dW1 - dW1) < 1e-8).all(), f"Expected result for dW1 is {expected_dW1}, but it returns {dW1}"
    #assert (abs(expected_db1 - db1) < 1e-8).all(), f"Expected result for db1 is {expected_db1}, but it returns {db1}"
    #
    #dW0, db0 = grads[0]
    #assert (abs(expected_dW0 - dW0) < 1e-8).all(), f"Expected result for dW0 is {expected_dW0}, but it returns {dW0}"
    #assert (abs(expected_db0 - db0) < 1e-8).all(), f"Expected result for db0 is {expected_db0}, but it returns {db0}"
