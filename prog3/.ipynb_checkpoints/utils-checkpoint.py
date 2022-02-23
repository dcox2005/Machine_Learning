##### >>>>>> Please put your name and 6-digit EWUID here  David Cox 284331


# Various tools for data manipulation. 



import numpy as np
import math

class MyUtils:

    
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        
        if degree == 1:
            return X
        
        features = len(X[0])
        bucketSize = [0] * (degree)
        
        for i in range(degree):
            bucketSize[i] = math.comb(i + features, features - 1)
        
        dPrime = np.sum(bucketSize)
        columnHeadValues = [0] * (dPrime)
        
        for i in range(features):
            columnHeadValues[i] = i
        
        previousBucketEnd = 0
        currentBucketSize = features
        currentWorkingColumn = currentBucketSize
        Z = X.copy()
        
        for i in range(1, degree):
            # print("i: ", i)
            for j in range(previousBucketEnd, previousBucketEnd + currentBucketSize):
                # print("j: ", j)
                headValue = columnHeadValues[j]
                # print("headValue: ", headValue)
                for k in range(headValue, features):
                    temp = (Z[:, j] * X[:, k]).reshape(-1, 1)
                    Z = np.append(Z, temp, axis = 1)
                    columnHeadValues[currentWorkingColumn] = k
                    currentWorkingColumn = currentWorkingColumn + 1

                
            previousBucketEnd = previousBucketEnd + currentBucketSize
            currentBucketSize = bucketSize[i]

            
        return Z
    
    def z_transform2(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        
        if degree == 1:
            return X
        
        n, features = X.shape
        bucketSize = [0] * (degree)
        
        for i in range(degree):
            bucketSize[i] = math.comb(i + features, features - 1)
        
        dPrime = np.sum(bucketSize)
        columnHeadValues = [0] * (dPrime)
        
        for i in range(features):
            columnHeadValues[i] = i
        
        previousBucketEnd = 0
        currentBucketSize = features
        currentWorkingColumn = currentBucketSize
        Z = np.zeros((n, dPrime), np.int32)
        Z[:n, :features] = X
        
        for i in range(1, degree):
            for j in range(previousBucketEnd, previousBucketEnd + currentBucketSize):
                headValue = columnHeadValues[j]
                for k in range(headValue, features):
                    temp = (Z[:, j] * X[:, k]).reshape(-1, 1)
                    Z[:n, currentWorkingColumn:currentWorkingColumn + 1] = temp
                    columnHeadValues[currentWorkingColumn] = k
                    currentWorkingColumn = currentWorkingColumn + 1
  
                    
                
            previousBucketEnd = previousBucketEnd + currentBucketSize
            currentBucketSize = bucketSize[i]

            
        return Z
    