from typing import List
from abc import ABC,abstractmethod

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



        #############################################################################
        # TODO: implement here the forward step.
        #
        # A few more instructions:
        # Note that the weights, bias and activation functions are class variables,
        # then you can access them via ``self``.
        #
        # You should store the input Z of each activation function in ``Z_list``,
        # and the output of the activation functions in ``activations``. These
        # information will be important when you implement the backward pass.
        #############################################################################
        return X


if __name__ == '__main__':
    m = Model([2, 1, 2], [ReLU(), ReLU()])

    X = np.array([
                   [0 ,1]
                  ,[-1,0]
                ])

    O = m.forward(X)

    import IPython; IPython.embed()
