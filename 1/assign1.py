#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import seaborn as sns

# meta data 
# 1. Frequency, in Hertzs.
# 2. Angle of attack, in degrees.
# 3. Chord length, in meters.
# 4. Free-stream velocity, in meters per second.
# 5. Suction side displacement thickness, in meters.
# The only output is:
# 6. Scaled sound pressure level, in decibels.
# print(df.head(10))


#airfoil_self_noise.dat
file_name= input("enter name of file")
epsilon=input("enter epsilon")



df = pd.read_csv(file_name, header=None, sep="\t",names=['Frequency', 'Angle_attack','Chord_len','FS_velocity','Suction_side_displ','SS_Pressure_level'])

print(type(df))
# print(df['Frequency'].mean())
df= np.array(df)
n,d=df.shape


# FINDING MEAN 
# axis=0 along columns , axis= 1 along rows 

mean_vector =np.sum(df, axis=0) /n
print("mean")
print(pd.DataFrame(mean_vector))
print("---------------")



# FINDIG  VARIANCE
#var_vector= np.var(df, axis=0)

center_df=[]
var_vector= []

# print("formula wala varrr", np.var(df, axis=0))
i=0
total_var=0
for x in df.T:
    temp=[]
    # find the inner summation for each column
    summ = (sum((xi - mean_vector[i])**2 for xi in x) /(n-1))
   
    temp=list((xi - mean_vector[i]) for xi in x)
   
    center_df.append(temp)
    i=i+1
    total_var+=summ
    
    var_vector.append(summ)
   
   
print("vairance")
print(pd.DataFrame(var_vector) )
print("total variance ", total_var,"\n")
center_df = np.array(center_df).T




     
# to find INNER covariance matrix
# dot products between the centered data matrix and it's transpose divided by total rows n 

inner_cov= np.dot(center_df.T ,center_df) /n
print("inner covarinace")
print(pd.DataFrame(inner_cov))


print("_________________")
#find OUTER covaience matrix
# dot product   then sum of all 
# TAKE ROW ONE , INVERSE IT AND DOT PRODUCT THE RESULT

for i in range(n):
    temp=np.array(center_df[i])
    # because otherwise it's giving (6,) instead of (6,1)  and asigning it to same variable gives error regarding shape
    z = temp.reshape((-1,1))
    if(i==0):
        
        outer_cov = np.dot(z, z.T) /n
        
    else:
        outer_cov+=np.dot(z, z.T) /n
        
       
     
print("outer covraince")
print(pd.DataFrame(outer_cov)  )


corr= np.empty(inner_cov.shape)

# zi is the column and so on ....  of center_df
# sigma 1-2 = z1. z2T and divided by root of z1T and z1
# find correlation with formula z1 z2t  and dividedd by l2 norm of z1 and z2
# check this one more time 


# transpose of center data
center_transpose =center_df.T
for i in range(center_transpose.shape[0]):
    for j in range(center_transpose.shape[0]):
        
        z1= center_transpose[i].reshape((-1,1))
        
        z2= center_transpose[j].reshape((-1,1))
        
        corr[i][j] = corr[j][i] = np.dot( z1.T / LA.norm(z1),z2 / LA.norm(z2))
        
print("____________")        
print("correlation matrix" ) 
print(pd.DataFrame(corr)  )
print("____________")        





# plt.matshow(corr)
# plt.show()
sns_plot = sns.pairplot(pd.DataFrame(corr))
sns_plot.savefig("scatter_plot.png")





#### scatter plots
#between  most correlated attributes 0 and 4
plt.scatter(center_df[:,0],center_df[:,4])
plt.show()





# most anti correlated 1 and 2
plt.scatter(center_df[:,1],center_df[:,2])
plt.show()





def power_iteration(A, e):
    
    
# Ideally choose a random vector
# To decrease the chance that our vector
# Is orthogonal to the eigenvector
    
    pi=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
    # making it orthogonal the second one
    
    pi[:,1]= pi[:,1]-(np.dot(pi[:,1].T,pi[:,0])/ LA.norm(pi[:,0]))*pi[:,0]
    
    flag=1
    
    p_k= pi
    while(flag):
        # dot product between xi and A
        p_k1= np.dot(A.T, p_k)
        # finding maximum value index
        indexes = np.where(p_k1 == np.amax(p_k1))    
         # finding max value 
        max_val= np.amax(p_k1)
       # calculating lambda eigen values
        eigen_values = p_k1[indexes[0]][indexes[1]]/ p_k[indexes[0]][indexes[1]]
        #normalizing the matroix by dividing it by max value 
        p_k1=np.divide(p_k1,max_val)
        # distance between X-1 and X 
        stopping_check = LA.norm(p_k1)-LA.norm(p_k)
        if(stopping_check <= float(e)):
            flag=0
            break
        else:
            # normalize p_k that is our vector is of unit length
            p_k1 = p_k1 / np.linalg.norm(p_k1)
            p_k=p_k1
                
            
            
    return p_k1,eigen_values 


# calling fucntion to get eigen vector and values, by passing covariance and epsilon 
vector_value = power_iteration(inner_cov, epsilon)

print("Eigen Values")
print(pd.DataFrame(vector_value[1]),'\n')
print("Eigen Vector")
print(pd.DataFrame(vector_value[0]))





new_plane_points= np.dot(center_df,vector_value[0])


plt.scatter(new_plane_points[:,0],new_plane_points[:,1])
plt.show()






