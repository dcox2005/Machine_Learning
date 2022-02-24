########## >>>>>> Put your full name and 6-digit EWU ID here. David Cox 284331

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        # remove the pass statement and fill in the code. 
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        n, d = np.shape(X)
        # tempArray = [0] * (len(X[0]) + 1)
        # self.w = np.array(tempArray).reshape(-1,1)
        self.w = np.zeros((d + 1,1))
        biasX = np.insert(X, 0, 1, axis=1)
        npY = np.array(y).reshape(-1,1)
        if (SGD):
            if (mini_batch_size > n or mini_batch_size < 1):
                mini_batch_size = n
            
            self.SGD(biasX, npY, eta, iterations, lam, mini_batch_size)
        else:
            # self.BGDVector(biasX, y, eta, iterations)
            self.BGDVectorReg(biasX, npY, eta, iterations, lam)
        
    
    def BGDVector(self, X, y, eta, iterations):
        n = len(y)
        step = eta / n
        for i in range(0, iterations):
            # part0 = (X @ self.w)
            # s = y * part0
            # part1 = LogisticRegression._v_sigmoid(-1 * s)
            # part2 = y * part1
            # part3 = np.transpose(X) @ part2
            # part4 = step * part3
            # self.w = self.w + part4
            
            s = (y * (X @ self.w))
            LR = LogisticRegression._v_sigmoid(-1 * s)
            self.w = self.w + step * (X.T @ (y * LR))
        
    
    def BGDVectorReg(self, X, y, eta, iterations, lam):
        n = len(y)
        step = eta / n
        for i in range(0, iterations):
            # s = y * (X @ self.w)
            # part1 = LogisticRegression._v_sigmoid(-1 * s)
            # part2 = y * part1
            # part3 = np.transpose(X) @ part2
            # part4 = step * part3
            # part5 = (2 * lam * eta) / n
            # part6 = 1 - part5
            # part7 = part6 * self.w
            # self.w = part7 + part4
            
            s = (y * (X @ self.w))
            LR = LogisticRegression._v_sigmoid(-1 * s)
            self.w = ((1 - ((2 * lam * eta) / n)) * self.w) + step * (X.T @ (y * LR))
            
            
    def SGD(self, X, y, eta, iterations, lam, mini_batch_size):
        
        # shuffler = np.random.permutation(len(y))
        # shuffledX = X[shuffler]
        # shuffledY = y[shuffler]
        shuffledX = X
        shuffledY = y
        n = len(y)
        # nPrime = mini_batch_size
        # step = eta / nPrime
        blockCount = math.ceil(n / mini_batch_size)
        
        for i in range(0, iterations):
            
            start = (i % blockCount) * mini_batch_size
            end = min(start + mini_batch_size, n)
            nPrime = end - start
            step = eta / nPrime
            
            yPrime = shuffledY[start: end, :]
            XPrime = shuffledX[start: end, :]
            # startingIndex = np.random.randint(0, n - mini_batch_size + 1)
            # yPrime = y[startingIndex: startingIndex + mini_batch_size, :]
            # XPrime = X[startingIndex: startingIndex + mini_batch_size, :]
            # s = yPrime * (XPrime @ self.w)
            # part1 = LogisticRegression._v_sigmoid(-1 * s)
            # part2 = yPrime * part1
            # part2a = np.transpose(part2)
            # part3 = part2a @ XPrime
            # part3a = np.transpose(part3)
            # part4 = step * part3a
            # part5 = (2 * lam * eta) / nPrime
            # part6 = 1 - part5
            # part7 = part6 * self.w
            # self.w = part7 + part4
            
            s = yPrime * (XPrime @ self.w)
            LR = LogisticRegression._v_sigmoid(-1 * s)
            self.w = (1 - ((2 * lam * eta) / nPrime)) * self.w + step * ((yPrime * LR).T @ XPrime).T
            
    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        # remove the pass statement and fill in the code. 
        X = MyUtils.z_transform(X, degree = self.degree)
        biasX = np.insert(X, 0, 1, axis=1)
        s = (biasX @ self.w)
        return LogisticRegression._v_sigmoid(s)
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        # remove the pass statement and fill in the code. 
        n = len(y)
        totalMissclassified = 0
        results = self.predict(X)
        for i in range(0,n):
            if(results[i] > 0.5):
                results[i] = 1
            else:
                results[i] = -1
            if (y[i] != results[i]):
                totalMissclassified += 1
        
        return totalMissclassified


    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

        # remove the pass statement and fill in the code. 
        vFunc = np.vectorize(LogisticRegression._sigmoid)
        var = vFunc(s)
        return var
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        # remove the pass statement and fill in the code.         
#         part1 = np.exp(-s)
#         part2 = 1 + part1
#         part3 = 1 / part2
#         return part3
    
        return (1 / (1 + np.exp(-s)))
    