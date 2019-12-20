import numpy as np
import sys
import pandas as pd
import math


def linear(A, B):
    return np.dot(A, B.T)

#  quadratic kernel hard coding different values of c

def linear_phi(A):
    return A

def quadratic(A, B):
     return (np.dot(A, B.T)) ** 2


def quadratic_phi(A):
    phi = pd.DataFrame()
    # to get xi2 of all the terms
    for i in range(A.shape[1]):
        temp = (A[:, i]) ** 2
        # np.hstack((phi,temp))
        phi.insert(loc=i, column="x"+str(i+1), value=temp)
    k = i
    # print(pd.DataFrame(phi))
    
    root2 = math.sqrt(2)
    for i in range(A.shape[1]):
        for j in range(i+1, A.shape[1], 1):
            temp = root2 * (A[:, i]) * A[:, j]
            phi.insert(loc=k, column="x"+str(i+1) + str(j+1), value=temp)
            k += 1
    # adding one column for bias
    temp = np.ones((n,1))
    phi.insert(loc=k, column="b", value=temp)
    # print(phi.columns)
    return phi

def guassian(A, B, spread):
    
    n1 = A.shape[0]
    n2 = B.shape[0]
    
    kernel = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            kernel[i][j] = math.exp(-(np.linalg.norm(A[i, :] -
                                                     B[j, :]))**2 / (2 * (float(spread) ** 2)))
    return kernel



#assign5.py TRAIN TEST C eps [linear OR quadratic OR gaussian ] spread
train = sys.argv[1]
test = sys.argv[2]
c = float (sys.argv[3])
eps = float (sys.argv[4])
kernel_type = sys.argv[5]
try:
    spread = sys.argv[6]
    print("For ", kernel_type, "kernel")

except IndexError:
    print("For ", kernel_type, "kernel")


train = pd.read_csv(train, header=None)
test = pd.read_csv(test, header=None)


train = np.array(train)
test = np.array(test)
# storing the shape of training and testing
n, d = train.shape
n_t, d_t = test.shape


# seperating x and y for traning data
train_x = train[:, 0: d-1]
train_y = train[:, d-1]

# seperating x and y for testing data
test_x = test[:, 0: d_t-1]
test_y = test[:, d_t-1]




# isko related kuch changes karne hai appending 1 wali cheez
#calcukate linear kernel
if (kernel_type == "linear"):
    kernel_matrix = np.add(linear(train_x, train_x), 1)
    kernel_test = np.add(linear(test_x, train_x), 1)
    all_one = np.ones((n, 1))
    train_x = np.hstack((train_x, all_one))
    phi =  train_x

elif (kernel_type == "quadratic"):
   
    kernel_matrix = np.add(quadratic(train_x, train_x), 1)
    kernel_test = np.add(quadratic(test_x, train_x), 1)
    phi = quadratic_phi(train_x)
    
    
    


elif kernel_type == "gaussian":
    kernel_matrix = np.add(guassian(train_x, train_x, spread,), 1)
    kernel_test = np.add(guassian(test_x, train_x, spread), 1)

 



# step size n_k
N_K= np.zeros((n,1))
for k in range(n):
    N_K[k] = float (1 / ( kernel_matrix[k][k])    )
   
# initializing alpha a
a_t = np.zeros((n,1))
t=0

a = a_t
a_t_1 = a_t

# αk ←αk + ηk 1−yk αiyiK(xi, xk)
while True:
    a = a_t
    for k in range(n):
        total_sum =0
        
        for i in range(n):
            
            total_sum += a[i]*train_y[i]* kernel_matrix[i][k]
        # update kth component of α
        
        a[k] = a[k] + N_K[k] * ( 1 - ( float (train_y[k]) * (float(total_sum[0])) ))
 
        if(a[k]< 0 ):
            a[k]  = 0
        if (a[k] > c):
            a[k] = c    
    a_t_1 = a
    t +=1

    if (np.linalg.norm(a_t_1 - a_t) <= eps):
        break
    else:
        a_t = a_t_1  


print("value of aplha grater than zero")
# print(pd.DataFrame(a_t_1))
count = 0
for i in range ( len(a_t_1)):
    if(a_t_1[i]>0):
        count += 1
        print(i, "  ", a_t_1[i])

print("number of Sv ==" , count)
# w  = sigma of ai yi and xi
# to compute w, we need feature space wale terms phii (x) wale 
# change this in feature
if (kernel_type =="linear"  or kernel_type == "quadratic"):
    phi = np.array(phi)
    w = np.zeros(phi.shape[1])
    for i in range(n):
        sigma = np.multiply(a_t_1[i]* train_y[i] , phi[i])

        w = np.add (  w , sigma )
     
    print("value of w is")
    print(w)



# print(n, n_t)
# print(kernel_test.shape)
# training accuracy

count = 0
for z in range(n):
    sigma = 0

    for i in range(n):
        if(a_t_1[i]>0):
            sigma += a_t_1[i] * train_y[i] * kernel_matrix[z][i]

    if(sigma > 0):
        pred_y = 1
    elif(sigma < 0):
        pred_y = -1
    else:
        pred_y = train_y[z]
        

    if (pred_y == train_y[z]):
        count += 1
print()
print("accuracy for training ", float((count*100) / n), "%")


count_bo = 0
# testing accuracy:
count = 0 
for z in range(n_t):
    sigma = 0

    for i in range(n):
        if(a_t_1[i]>0):
            #sigma += np.multiply(a_t_1[i] * train_y[i], kernel_test[z][i])
            sigma += a_t_1[i] * train_y[i] * kernel_test[z][i]


    if(sigma>0):
        pred_y = 1
    elif(sigma<0):
        pred_y = -1
    else:
        pred_y = test_y[z]
        count_bo += 1

    if (pred_y == test_y[z]):
        count += 1

print(count_bo)
print("accuracy for testing ", float((count*100)/ n_t), "%")

