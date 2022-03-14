# Place your EWU ID and Name here. 

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



# import numpy as np
# import math
# import math_util as mu
# import nn_layer

from math import sqrt
import numpy as np
import math
from code_NN.nn_layer import NeuralLayer as nn_layer
import code_NN.math_util as mu
import code_NN.nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        
        # newLayer =  nn_layer.NeuralLayer(d, act)
        newLayer =  nn_layer(d, act)
        self.layers.append(newLayer)
        self.L = self.L + 1

        
        
    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        
        weight_rng = np.random.default_rng(2142)
        for l in range(1, len(self.layers)):
            low = (-1 / np.sqrt(self.layers[l].d))
            high = (1 / np.sqrt(self.layers[l].d))
            size = ((self.layers[l - 1].d + 1), (self.layers[l].d))
            self.layers[l].W = weight_rng.uniform(low, high, size)
           
        
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        
        n = len(X)
        blockCount = math.ceil(n / mini_batch_size)
        for i in range(0, iterations):
            start = (i % blockCount) * mini_batch_size
            end = min(start + mini_batch_size, n)
            nPrime = end - start
            step = eta
            
            # XPrime = X
            # YPrime = Y
            XPrime = X[start: end, :]
            YPrime = Y[start: end, :]

            self._batchFit(XPrime, YPrime, nPrime, step)
        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

    
    def _batchFit(self, X, Y, nPrime, step):
        self._forwardFeed(X)
        lastLayer = self.layers[self.L]
        # deltaPart1 = lastLayer.X[:,1:] - Y
        # deltaPart2 = lastLayer.act_de(lastLayer.S)
        # lastLayer.delta = 2 * deltaPart1 * deltaPart2
        lastLayer.delta = 2 * ((lastLayer.X[:,1:] - Y) * lastLayer.act_de(lastLayer.S))
        lastLayer.G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X, lastLayer.delta) * (1 / nPrime)
        self._backPropagation(nPrime)
        self._updateWeights(step)
        
    
    def _forwardFeed(self, X):
        biasX = np.insert(X, 0 , 1, axis = 1)
        self.layers[0].X = biasX
        for l in range(1, self.L + 1):
            self.layers[l].S = self.layers[l - 1].X @ self.layers[l].W
            activated = self.layers[l].act(self.layers[l].S)
            self.layers[l].X =  np.insert(activated, 0, 1, axis = 1)

            
        
    def _backPropagation(self, nPrime):
        for l in range(self.L - 1, 0, -1):
            currentLayer = self.layers[l]
            forwardLayer = self.layers[l + 1]
            # deltaPart0 = forwardLayer.W[1:,:]
            # deltaPart1 = deltaPart0.T
            # deltaPart2 = forwardLayer.delta @ deltaPart1
            # deltaPart3 = currentLayer.act_de(currentLayer.S) * deltaPart2
            # currentLayer.delta = deltaPart3
            currentLayer.delta = currentLayer.act_de(currentLayer.S) * (forwardLayer.delta @ (forwardLayer.W[1:,:]).T)
            currentLayer.G = np.einsum('ij, ik -> jk', self.layers[l- 1].X, currentLayer.delta) * (1 / nPrime)
        
    
    def _updateWeights(self, step):
        for l in range(1, self.L + 1):
            currentLayer = self.layers[l]
            part1 = step * currentLayer.G
            part2 = currentLayer.W - part1
            currentLayer.W = part2
        
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''        
        self._forwardFeed(X)
        nonBias = self.layers[self.L].X[:,1:]
        prediction = np.argmax(nonBias, axis=1).reshape(-1, 1)
        
        return prediction
    
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        n = len(Y)
        prediction = self.predict(X)
        yIndex = np.argmax(Y, axis=1)
        y = yIndex.reshape(-1, 1)
        return (np.sum(prediction != y) / n)
    