import numpy as np
import numpy.linalg as la
# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
#         numpy matrix Z, with d rows, F columns
def run(F,X):
    X = np.copy(X)
    (row ,cols)=np.shape(X)
    mu = np.zeros((cols,1))
    for i in range(0, cols):
        total = 0
        for t in range(row):
            total = total + X[t][i]
        mu[i] = total/row
    for t in range(0, row):
        for i in range(0, cols):
            X[t][i] = X[t][i] - mu[i]
    U, s, VT = la.svd(X, False)
    g = np.zeros(F)
    for i in range(0, F):
        if (s[i] > 0):
            g[i] = 1 / s[i]
    W = np.zeros((F,cols))
    for i in range(0, F): #first F rows of matrix VT
        W[i] = VT[i]
    Z = np.dot(W.T, np.diag(g))
    return (mu, Z)

