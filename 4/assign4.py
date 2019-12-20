# assign4.py TRAIN TEST m η epochs=maxiter

import random
import numpy as np
import sys
import pandas as pd
import math
import random
from operator import add

train = sys.argv[1]
test = sys.argv[2]
m = int(sys.argv[3])
learning_rate = float  (sys.argv[4])
epochs = int(sys.argv[5])


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

# max number y  in train_y will be the  number required for hot encoding
unique = list(np.unique(train_y))
unique_len = len(unique)
print(unique)


hot_encoded_y  = []

# to hot encode the response variable - training
for  i in range(len(train_y)):
    temp = [0]* unique_len
    temp [ unique.index(train_y[i])] = 1
    hot_encoded_y.append(temp)
    if(temp == 1):
        print(temp)
# to hot encode the response variable - testing
hot_encoded_y = np.array ( hot_encoded_y)
hot_encoded_y_test = []
for i in range(len(test_y)):
    temp = [0] * unique_len
    temp[unique.index(test_y[i])] = 1
    hot_encoded_y_test.append(temp)
hot_encoded_y_test = np.array (  hot_encoded_y_test)

# generating random order 
r = list(range(n))
random.shuffle(r)
# p is the number of variables that are unique K 
p= unique_len


def MLP_train(D, m, learning_rate, epochs):
    # initialise bias
    b_h = pd.DataFrame([0.5] * m)
  
    b_o = pd.DataFrame([0.5] * p)
    # initialise weigth vector 
    w_h =np.array([[0.1 for col in range(m)]
              for row in range(d-1)], dtype=float)
    
    w_o = np.array([[0.1 for col in range(p)]
                    for row in range(m)], dtype=float)
    

    t =0

    # calculating the net at the output layer
    
    while(t<epochs):

        
        # random iteration
        for i in r:
            # feed forward
            # this is for calculating the net z values - hidden layer

            net_z = np.array (b_h +  np.dot(w_h.T, D[i]).reshape((-1,1)) )

            for j in range(len(net_z)):
                net_z[j] = float( max(0,net_z[j]))

            # to calculate next layer z to o
            #SOFTMAX
            net_o = np.array(b_o + np.dot(w_o.T, net_z))
            # ab isko softmax me do 
            
            total =0
            for j in range(len(net_o)):
                total += math.exp(net_o[j][0])
            for j in range(len(net_o)):
                net_o[j][0] = math.exp(net_o[j][0])   
            o = np.divide ( net_o,  total) # chanhge this 



            #   Backpropagation phase: net gradients
            # it derivative of relu with elemtn wise product of wh and delta o at the output
            derivative_relu =[0]  * len(net_z)
            for j in range(len(net_z)):
                if(net_z[j] > 0):
                    derivative_relu[j] = 1
                
           
            delta_o = np.array([ y-x for x, y in zip(hot_encoded_y[i], o)])
            # print("the reuiqed chapes " , type (derivative_relu), w_o.shape, np.array( delta_o.shape))
            delta_h =np.array ( [ sum(x) for x in zip(
                list(derivative_relu), list (np.dot(w_o, delta_o))) ])


            #Gradient descent for bias vectors
            delta_bo = delta_o
            b_o = b_o- learning_rate * delta_bo
            delta_bh = delta_h
            
            b_h = b_h - learning_rate * delta_bh

            #Gradient descent for weight matrices
            delta_wo = np.dot(net_z,delta_o.T)
            w_o = w_o - learning_rate* delta_wo
           
            delta_wh = np.dot(delta_h, D[i].reshape((1,d-1)))
            w_h = w_h - learning_rate * delta_wh.T

        print("iteration", t)
        t+=1
        
         

    # print("im done ")
    return w_h,w_o, b_h,b_o


    

w_h, w_o,b_h, b_o = MLP_train(train_x, m, learning_rate, epochs)
# printing the required variuables
print("b hidden")
print(pd.DataFrame(b_h))

print("w hidden")
print(pd.DataFrame(w_h))

print("b output")
print(pd.DataFrame(b_o))

print("w output")
print(pd.DataFrame(w_o))

# calcukating accuracy for training dataset
count = 0

for i in range(n):
    
    net_z = np.array(b_h + np.dot(w_h.T, train_x[i]).reshape((-1, 1)))
    
    
    for j in range(len(net_z)):
         net_z[j] = float(max(0, net_z[j]))

            #SOFTMAX
   
   
    net_o = np.array(b_o + np.round( np.dot(w_o.T, net_z), 9))
         
    
    total = 0
    
    for j in range(len(net_o)):
        total += math.exp(net_o[j][0])
           
    o = np.divide(net_o,  total) 
    
    # we hve to change this to  So in the 
    #  one-hot y_hat vector do we take the largest number as 1 and the rest 0 to compute the accuracy?
    check = []
    for j in range(len(o)):
        check.append(float(o[j]))
    check = np.array (check)   
    max_val =  max(check)
    for j in range(len(check)):
        if ( check[j] == max_val):
            check[j]= 1
        else :
            check[j] =0 
    
    if(np.array_equal (check, hot_encoded_y[i])):
        count+=1
          

print(" training accuracy = ", float ( count/n) *100 ,"%")


# calcukating accuracy for testing dataset
count =0
for i in range(n_t):
    # calculating net z 
    net_z = np.array(b_h + np.dot(w_h.T, test_x[i]).reshape((-1, 1)))
    
    # applying relu fucntion 
    for j in range(len(net_z)):
         net_z[j] = float(max(0, net_z[j]))

            #SOFTMAX
   
   # calculating net o 
    net_o = np.array(b_o + np.round( np.dot(w_o.T, net_z), 9))
         
    # applying softmax 
    total = 0
    # this net_on  is coming same for every case though net_z is different 
    for j in range(len(net_o)):
        total += math.exp(net_o[j][0])
           
    o = np.divide(net_o,  total)  # chanhge this
    
    # we hve to change this to  So in the 
    #  one-hot y_hat vector do we take the largest number as 1 and the rest 0 to compute the accuracy?
    # chaning o - output so that we can compare it with the test_y
    check = []
    for j in range(len(o)):
        check.append(float(o[j]))
    check = np.array (check)   
    max_val =  max(check)
    for j in range(len(check)):
        if ( check[j] == max_val):
            check[j]= 1
        else :
            check[j] =0 
    
    if(np.array_equal (check, hot_encoded_y_test[i])):
        count+=1
  

print(" testing accuracy = ", float ( count/n_t) *100 ,"%")




