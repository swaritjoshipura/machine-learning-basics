import numpy as np
# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
#        numpy vector mu, with d rows, 1 column
#        numpy matrix Z, with d rows, F columns
# Output: numpy matrix P, with n rows, F columns
def run(X,mu,Z):
    X = np.copy(X)
    (n ,d)=np.shape(X)
    for t in range(0, n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]
    P = np.dot(X, Z)
    return P