##### >>>>>> Please put your name and 6-digit EWUID here


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
        
        Z = X.copy()   #you don't want to mess with the original data
        
        r = degree   #how many buckets do you have
        n, d = X.shape  # n is rows, d is columns
        B = []
        for i in range(r):
            B.append(math.comb(d + i, d - 1)) # n choose k of the ith bucket     # no i - 1 because of zero based indexing, if using 1 based indexing, minus 1
        
        ell = np.arange(np.sum(B))   #arange produces numbers starting at 0 and going through the sum of B array
        q = 0   # the total size of all buckets befroe the previous bucket
        p = d   # the size of the previous bucket
        for i in range(1, r): # range does not include r, so it is r-1
            for j in range(q, q + p):
                headNumber = ell[j]  #this is the number that floats above each column which is the highest rated feature in this column
                for k in range(headNumber, d):  # this will go through all indexes that match the head above, and go through that and all greater.
                    temp = (Z[:, j] * X[:, k]).reshape(-1, 1)    #the : indicates all rows. So this is all rows in the jth column
                    Z = np.append(Z, temp, axis = 1)
                    ell[j] = k
                    
                
            q = q + p
            p = B[i]
        
        assert Z.shape[1] == np.sum(B)
        return Z
    
    
    