import numpy as np
import math
import numpy.linalg as la


# Input: scalar q
#        numpy vector mu_positive of d rows, 1 column
#        numpy vector mu_negative of d rows, 1 column
#        scalar sigma2_positive
#        scalar sigma2_negative
#        numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(q, mu_positive, mu_negative, sigma2_positive, sigma2_negative, z):
    a = math.log(q / (1 / q))
    b = (mu_positive.shape[0] / 2) * math.log(sigma2_positive / sigma2_negative)
    c = 1 / (2 * sigma2_positive) * (la.norm(z - mu_positive) ** 2)
    d = 1 / (2 * sigma2_negative) * (la.norm(z - mu_negative) ** 2)

    if a - b - c + d > 0:
        label = 1
    else:
        label = -1
    return label
