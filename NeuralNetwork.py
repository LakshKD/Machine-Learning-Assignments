# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 02:32:09 2016

@author: laksh
"""
import numpy as np
import csv
import math as d
def sigmoidal_function(w,x):
    s = np.dot(w,x)
    if s < -100:
        return 0
    elif s>100:
        return 1
        
    
    y = 1.0 + d.exp(-1.0 * s)    
    Output = 1.0 / y
    if(Output>=0.5):
        return 1
    else:
        return 0


data = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 58))
Train_actual_output = [row[57] for row in data]
a = np.ones(3000)
Train_feature_vector = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 57))

X = np.column_stack((Train_feature_vector,a)) 
'''print feature_vector.shape'''
W = np.ones(58)


alpha = 0.01
j = 1
while j<=25000:    
    for i in range(0,3000):
        o = sigmoidal_function(W,X[i])
        if(o ==0 and o != Train_actual_output[i]):
            '''W_new = W'''
            W = np.add(W,np.dot(alpha,X[i]))
        else :
            if(o ==1 and o != Train_actual_output[i]):
                '''W_new = W'''
                W = np.subtract(W,np.dot(alpha,X[i]))
    j = j+1
#print W

Test_feature_vector = np.loadtxt('TestX.csv', delimiter=',', usecols=range(0, 57))
b = np.ones(1600)
p = np.zeros(1600)
Test_feature_vector = np.column_stack((Test_feature_vector,b)) 
for i in range(0,1600):
         p[i] = sigmoidal_function(W,Test_feature_vector[i])
        
with open('Neural_Network.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(0,1600):
        writer.writerow([i,int(p[i])])

    
              



'''print feature_vector.shape'''

