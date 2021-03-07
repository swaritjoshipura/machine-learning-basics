# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#           y[i] is the label (+1 or -1) of the i-th sample
#
#
#
# Output: scalar q
#        numpy vector mu_positive of d rows, 1 column
#        numpy vector mu_negative of d rows, 1 column
#        scalar sigma2_positive
#        scalar sigma2_negative
import numpy as np
import numpy.linalg as la



def run(X, y):
    kp = 0
    kn = 0
    mu_positive = 0
    mu_negative = 0

    for t in range(0, len(X)):
        if y[t] == 1:
            kp = kp + 1
            mu_positive = mu_positive + X[t]
        else:
            kn = kn + 1
            mu_negative += X[t]
    q = kp / len(X)
    mu_positive = (1 / kp) * mu_positive
    mu_negative = (1 / kn) * mu_negative
    sigma2_positive = 0
    sigma2_negative = 0
    for t in range(0, len(X)):
        if y[t] == 1:
            sigma2_positive = sigma2_positive + (la.norm(X[t] - mu_positive) ** 2)
        else:
            sigma2_negative = sigma2_negative + (la.norm(X[t] - mu_negative) ** 2)
    sigma2_positive = (1 / (X.shape[1] * kp)) * sigma2_positive
    sigma2_negative = (1 / (X.shape[1] * kn)) * sigma2_negative

    return q, (np.c_[mu_positive]), (np.c_[mu_negative]), sigma2_positive, sigma2_negative
