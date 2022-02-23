# >>>>>>>>>>>>> Please place your full name and 6-digit EWU ID here: David Cox 284331


# Implementation of the perceptron learning algorithm. Support the pocket version for linearly unseparatable data. 



# NOTE: Your code need to vectorize any operation that can be vectorized. 


import numpy as np

class PLA:
    def __init__(self):
        self.w = None
        
    def fit(self, X, y, pocket = True, epochs = 100):
        ''' X: n x d matrix, representing n samples, each has d features. It does not have the bias column. 
            y: n x 1 matrix of {+1, -1}, representing the n labels of the n samples in X.

            return: self.w, the (d+1) x 1 weight matrix representing the classifier.  

            1) if pocket = True, the function will run the pocket PLA and will update the weight
               vector for no more than epoch times.  

            2) if pocket = False, the function will run until the classifier is found. 
               If the classifier does not exist, it will run forever.             
        '''
        
        ### add your code here. 
        
        # Hint: In practice, one will scan through the entire set to find the next misclassied sample. 
        #       One will repeatedly scan the data set until one scan does not see any misclassified sample. 
        
        #       Use matrix/vector operation in checking each training sampling. 
        
        #       In the pocket version, you can use the self.error function you will develop below. 
        
        #take X and add new column of all 1
        
        
        # y' = x @ w
        # np.sum(y'== y)

        tempArray = [0] * (len(X[0]) + 1)
        self.w = np.array(tempArray).reshape(-1,1)
        biasX = np.insert(X, 0, 1, axis=1)
        numberOfRuns = 0
        updated = True
        wStar = self.w
        misClassifiedSelfW = len(biasX)
        while (pocket and updated and (numberOfRuns < epochs)):
            updated = False
            for i in range(len(biasX)):
                misclassifiedCurrent = 0
                if (np.sign(biasX[i] @ wStar) != y[i]):
                    updated = True
                    wStar = wStar + (y[i] * biasX[i]).reshape(-1,1)
                    for j in range(len(biasX)):
                        if (np.sign(biasX[j] @ wStar) != y[j]):
                            misclassifiedCurrent = misclassifiedCurrent + 1
                            
                    if (misclassifiedCurrent < misClassifiedSelfW):
                        self.w = wStar
                        misClassifiedSelfW = misclassifiedCurrent  
                        
                        
                        
            numberOfRuns = numberOfRuns + 1

        while (updated and not pocket):
            updated = False
            for i in range(len(biasX)):
                if (np.sign(biasX[i] @ self.w) != y[i]):
                    updated = True
                    self.w = self.w + (y[i] * biasX[i]).reshape(-1,1)
        
        return self.w
            
        
    def predict(self, X):
        ''' X: n x d matrix, representing n samples and each has d features, excluding the bias feature. 
            return: n x 1 vector, representing the n labels of the n samples in X. 
            
            Each label could be +1, -1, or 0. 
            
            Note: We let the users to decide what to do with samples 
                  that sit right on the classifier, i.e., x^T w = 0
        '''

        ### add your code here 
        
        # Hint: use matrix/vector operation to predict the labels of all samples in one shot of code. 
        
        #take x and matrix multiply with w, apply the sign function to get 1,-1
        #x * self.w = matrix apply sign function to get vector of 1, -1
        
        biasX = np.insert(X, 0, 1, axis=1)
        prediction = [0] * len(biasX)
        for i in range(len(biasX)):
            prediction[i] = np.sign(biasX[i] @ self.w)
        
        prediction = prediction.reshape(-1,1)
        return prediction
        
    
    def error(self, X, y):
        ''' X: n x d matrix, representing n samples and each has d features, excluding the bias feature.
            y: n x 1 vector, representing the n labels of the n samples in X. Each label is +1 or -1. 
            
            return: the number of samples in X that are misclassified by the classifier
            
            Note: we count a sample x that sits right on the classifier, x^T w = 0, as a misclassified one. 
        '''
        
        # add your code here
        
        # Hint: use matrix/vector operation to get predicated label vector in one shot of code. 
        #       Then use vector comparison to compare the given label vector and 
        #       the predicted label vector, along with the help from the numpy.sum function
        #       to count the #misclassified quickly. 
        
        #
        misclassified = 0
        biasX = np.insert(X, 0, 1, axis=1)
        for j in range(len(biasX)):
            if (np.sign(biasX[j] @ self.w) != y[j]):
                misclassified = misclassified + 1
                
        return misclassified
    
