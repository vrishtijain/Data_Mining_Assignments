import numpy as np
import sys
import pandas as pd
import math



def sigmoid(z):
    return float(1/ (1+math.exp(-z)))

# reading command line arguments
train = sys.argv[1]
test = sys.argv[2]
eps = sys.argv[3]
eta = sys.argv[4]

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




weight_vector = [0] * d
weight_vector = np.array(weight_vector).reshape((-1,1))
weight_t =weight_vector
weight_t_1 = weight_vector


# logistic regression algo
while True:
    weight_next = weight_t
    for i in range(n):
        
        x = np.array(train_x.iloc[i, :]).reshape((-1,1))
        
        theta_z = sigmoid  (np.dot(weight_next.T, x)[0]) 
        
        del_wi_xi = np.dot(float (train_y[i] - theta_z ),x)
        # print(del_wi_xi.shape, "shape ofn eidfoi", weight_next.shape)
        del_wi_xi = np.array(del_wi_xi).reshape((-1, 1))
        weight_next = np.add (weight_next , np.dot(del_wi_xi ,float(eta)))


    weight_t_1 = weight_next
        
    
    check = np.linalg.norm(weight_t_1 - weight_t)
   # break when value of check becomes less than eps
    if(check <= float(eps)):
        # print("isnide this")
        break
    else :
        weight_t = weight_t_1

print("weight vector ",weight_t_1)    


#try on testing data  find y 
n_test, d_test = test_x.shape


# calculating the accuracy for testing dataset
count = 0
incorrect =0
for i in range(n_test):
    # calculate thee=ta of wixi 
    theta_z = sigmoid(np.dot((weight_t_1.T), test_x[i, :]))
    if( float (theta_z ) >= 0.5):
        pred = 1
    else :
        pred  =  0

    if( pred == test_y[i] ):
        count +=1  
    else:
        incorrect+=1


print(count, incorrect , n_test)
print("accuracy is " , float ( count / n_test))






