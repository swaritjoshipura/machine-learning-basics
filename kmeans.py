import numpy as np
import numpy.linalg as la
# Input: number of iterations L
# Output: numpy vector a of n rows, 1 column
#            a[i] is the cluster assignment for the i-th sample (an integer from 0 to k-1)
#        number of iterations that were actually executed (iter+1)
def run(L,k,X):
  n,d = np.shape(X) 
  a = np.zeros(n) 
  r = np.copy(X)
  q = np.random.choice(n,k,replace = False)
  for j in range(0, k):
    i = q[j]
    r[j] = X[i] 
  for iter in range(0,L):
    at_least_one_change = False
    for t in range(n):
      c = 0
      b = la.norm(np.c_[X[t]] - np.c_[r[0]])
      for j in range(k):
        if la.norm(np.c_[X[t]] - np.c_[r[j]]) < b:
          c = j
          b = la.norm(np.c_[X[t]] - np.c_[r[j]])
      if a[t] != c:
        a[t] = c
        at_least_one_change = True
    if at_least_one_change == False:
      break
    for j in range(0,k):
      s = np.zeros(d)
      m = 0
      for t in range(0,n):
        if a[t] == j:
          s = s + X[t]
          m = m + 1
      r[j] = s/m
  return np.c_[a].astype(int), iter+1