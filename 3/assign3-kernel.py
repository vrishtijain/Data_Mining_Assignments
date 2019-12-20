import numpy as np
import sys
import pandas as pd
import math


def linear (A,B):
    return np.dot(A,B.T)

# inhomogerous quadratic kernel hard coding different values of c

def quadratic(A, B):
     return (5 + np.dot(A, B.T)) ** 2

def guassian(A,B, spread):
    
    n1 = A.shape[0]
    n2= B.shape[0]
    kernel = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            kernel[i][j]= math.exp(-(np.linalg.norm(A.iloc[i,:] - B.iloc[j,:])**2) / (2 * (float(spread) ** 2)))
    return kernel
            

    


train = sys.argv[1]
test = sys.argv[2]
kernel_type = sys.argv[3]
try:
    spread = sys.argv[4]
    print("For ", kernel_type, "kernel")
    
except IndexError:
    print("For " , kernel_type, "kernel")
    
        


alpha=0.01

train = pd.read_csv(train, header=None)
test = pd.read_csv(test, header=None)

n, d = train.shape
n_test, d_test = test.shape

x_0 = pd.DataFrame([1] * n)

#train_y = train.iloc[:, n-1]

#  seperating y column
train_y = train.iloc[:, d-1]

test_y = test.iloc[:, d-1]
# dropping y from train
train_x = train.drop(train.columns[d-1], axis=1)
test_x = test.drop(test.columns[d-1], axis=1)



#calcukate linear kernel
if (kernel_type == "linear"):
    
    kernel_matrix = np.add(linear(train_x, train_x), 1)
    kernel_test = np.add(linear(test_x, train_x), 1)

elif (kernel_type =="quadratic"):
    kernel_matrix = np.add(quadratic(train_x, train_x), 1)
    kernel_test = np.add(quadratic(test_x, train_x), 1)


elif kernel_type == "gaussian":
    
    kernel_matrix = np.add(guassian(train_x, train_x, spread,), 1)
    kernel_test = np.add(guassian(test_x, train_x, spread), 1)


else:
    print("invalid kernel type")
    sys.exit()



# kernal ridge algorithm
kernel_matrix = np.array(kernel_matrix)
alpha_identity = np.dot ( alpha , np.identity(n)  )
c = np.dot (np.linalg.inv (kernel_matrix + alpha_identity), train_y)
y_pred = np.dot (kernel_matrix, c )



# calculating the accuracy for training dataset
count =0
for i in range(n):
    if(y_pred[i] >= 0.5):
        pred = 1
    else:
        pred =0    
    if(pred == train_y[i]):
        count+= 1

print("training accuracy :", float(count/ n )*100 ,"%")




# testing the algo


y_test_pred = np.dot(kernel_test, c)
# calculating the accuracy for testing dataset
count = 0
incorrect =0
for i in range(n_test):
    if(y_test_pred[i] >= 0.5):
        pred = 1
    else:
        pred = 0
    if(pred == test_y[i]):
        count += 1
    else:
        incorrect+=1    
# print(count, incorrect, n_test)
print("testing accuracy :", float(count / n_test)*100, "%")











