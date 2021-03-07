import numpy as np
# Input: maximum number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (+1 or -1) of the i-th sample

# Output: numpy vector theta of d rows, 1 column
#        number of iterations that were actually executed (iter+1)
def run(L,X,y):
    (rows,cols)=np.shape(X)
    theta = np.zeros((cols, 1))
    for i in range(0, L):
        all_points_classified_correctly = True
        for t in range(0, rows):
            if (y[t]*(np.dot(X[t],theta))[0]) <= 0:
                theta = theta + np.array([y[t]* X[t]]).T
                all_points_classified_correctly = False
        if all_points_classified_correctly:
            break
    return theta, i+1
