########## >>>>>> Put your full name and 6-digit EWU ID here.  David Cox 284331


# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 



import numpy as np
import math
import sys
sys.path.append("..")

from misc.utils import MyUtils


class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1
        
        
    def fit(self, X, y, CF = True, lam = 0, eta = 0.01, epochs = 1000, degree = 1):
        ''' Find the fitting weight vector and save it in self.w. 
            
            parameters: 
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        '''
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        
        if CF:
            self._fit_cf(X, y, lam)
        else: 
            self._fit_gd(X, y, lam, eta, epochs)
 


            
    def _fit_cf(self, X, y, lam = 0):
        ''' Compute the weight vector using the clsoed-form method.
            Save the result in self.w
        
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        ## delete the `pass` statement below.
        ## enter your code here that implements the closed-form method for
        ## linear regression 
        
        biasX = np.insert(X, 0, 1, axis=1)
        XT = np.transpose(biasX)
        #self.w = (np.linalg.pinv((XT @ biasX)) @ (XT @ y))
        Identity = np.identity(len(X[0]) + 1)
        self.w = (np.linalg.pinv((XT @ biasX + (lam * Identity))) @ (XT @ y))

    
    
    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        ## delete the `pass` statement below.
        ## enter your code here that implements the gradient descent based method
        ## for linear regression 
        tempArray = [0] * (len(X[0]) + 1)
        self.w = np.array(tempArray).reshape(-1,1)

        biasX = np.insert(X, 0, 1, axis=1)
        XT = np.transpose(biasX)
        n = len(X)
        
        Identity = np.identity(len(X[0]) + 1)
        movement = ((2 * eta) / n)
        #part1 = (Identity - (movement * (XT @ biasX)))
        part1 = (Identity - (movement * (XT @ biasX + (lam * Identity))))
        part2 = (movement * (XT @ y))
        
        for i in range(0, epochs):
            self.w =  part1 @ self.w + part2
        
    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''

        ## delete the `pass` statement below.
        
        ## enter your code here that produces the label vector for the given samples saved
        ## in the matrix X. Make sure your predication is calculated at the same Z
        ## space where you trained your model. 

        X = MyUtils.z_transform(X, degree = self.degree)
        biasX = np.insert(X, 0, 1, axis=1)
        return (biasX @ self.w)
        

    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''

        ## delete the `pass` statement below.
        ## enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix x
        ## Make sure your predication is calculated at the same Z space where you trained your model. 

        n = len(X)
        X = MyUtils.z_transform(X, degree = self.degree)
        biasX = np.insert(X, 0, 1, axis=1)
        return (np.sum(pow((biasX @ self.w) - y, 2)) / n)


