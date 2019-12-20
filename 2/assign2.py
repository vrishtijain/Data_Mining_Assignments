import numpy as np
import sys
import pandas as pd

# check for dimension match 
train = sys.argv[1]
test = sys.argv[2]
ridge = sys.argv[3]


train = pd.read_csv(train, header=None)
test = pd.read_csv(test, header=None)
n, d = train.shape

x_0 = pd.DataFrame([1] * n)

#train_y = train.iloc[:, n-1]

#  seperating y column
train_y = train.iloc[:, d-1]

test_y = test.iloc[:, d-1]
# dropping y from train
train_x = train.drop(train.columns[d-1], axis=1)
test_x = test.drop(test.columns[d-1], axis=1)

# # normalizing the train_x
mean_x = np.mean(train_x, axis=0)
mean_y = np.mean(train_y, axis=0)

mean_y_test = np.mean(test_y, axis=0)
# center_train = train_x - mean
# inserting vector x_0 with all 1s
train_x.insert(loc=0, column="x_0", value=x_0)
test_x.insert(loc=0, column="x_0", value=x_0)

# initilize R


def QR(train_x, train_y, test_x, test_y, ridge):
        # add ridge factored rows at the end as well
        A = np.zeros((d, d))
        # reshaping train_y becuase it was (35064,) --> (35604,1)
        train_y = np.array(train_y)
        train_y = train_y.reshape((-1, 1))
        
        
        add_in_y= np.zeros((d,1))
        
       # A vector with diagonal elemnets as sqrt ridge value
        for i in range(d):
            A[i][i] = float(ridge)**(1/2)
        # appending A with train_X
        train_x = np.vstack((train_x, A))

        # adding zeroes in train_y vector as well
        train_y = np.vstack((train_y, add_in_y))
        # new shapes of n_X and n_y
        n_x, d_x = train_x.shape
        # initializing R vector
        R = np.zeros((d_x, d_x))
        for i in range(d_x):
                R[i][i] = 1
        

        Q = np.array([[0 for col in range(d_x)]
                      for row in range(n_x)], dtype=float)

# q r factorization code:
        Q[0:, 0] = 1
        for x in range(1, d_x):

                X = train_x[:, x]
                Q[:, x] = X

                for i in range(x):
                        R[i][x] = (np.dot(X.T, Q[:, i])) / \
                            (np.linalg.norm(Q[:, i])**2)
                        Q[:, x] -= R[i][x] * Q[:, i]

        delta = np.array([[0 for col in range(d_x)]
                          for row in range(n_x)], dtype=float)

        for i in range(d_x):
            
                delta[i][i] = float(1 / np.linalg.norm(Q[:, i])**2)

#Rw˜ = delta QTY
        
        temp = np.array(train_y).reshape((-1, 1))
        # RW calculated
        r_w = np.dot(np.dot(delta, Q.T), temp)
        #initializing w_vector
        w_vector = [0] * d_x

# back substitution to get w_vector
        for j in range(d_x-1, -1, -1):
                w_vector[j] = r_w[j]
                for i in range(j+1, d_x, 1):
                        w_vector[j] -= R[j][i] * r_w[i]

        print("values of w_vector ")

        print(pd.DataFrame(w_vector))
        w_vector = np.array(w_vector)
        print("L2 norm so weight vector ", np.linalg.norm(w_vector))
        sse = 0

# sum of sq error for training data set

        for i in range(n_x):
            # calculatin y = w1a1+ w2a2......
                y_array = np.dot(train_x[i, :].T, w_vector)

                sse += (train_y[i] - y_array)**2

        print("sse for training", sse)
        tss = 0
# calculate tss
        for i in train_y:
                tss += (i-mean_y)**2

        r2 = (tss-sse) / tss
        print("r2 for training ", r2)

        t_sse = 0
        t_tss = 0
# sum of sq error for testing data set
        for i in range(n):
            # calculatin y = w1a1+ w2a2......
                y_array = np.dot(test_x.iloc[i, :].T, w_vector)
                t_sse += (test_y[i] - y_array)**2

        print("sse for test", t_sse)

# calculate tss

        for i in test_y:
                t_tss += (i-mean_y_test)**2

# print(tss)
        t_r2 = (t_tss-t_sse) / t_tss
        print("r2 for test", t_r2)

# QR function called to calculate 
QR(train_x, train_y, test_x, test_y,ridge)
