# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 00:37:53 2016

@author: laksh
"""
import numpy as np
import csv
import math as mp

def sigmoidal_function(w,x):
    s = np.dot(w,x)
    y = 1.0 + mp.exp(-1.0 * s)    
    Output = 1.0 / y
    return Output
        
def thresh_sigmoidal_function(w,x):
    s = np.dot(w,x)
    y = 1.0 + mp.exp(-1.0 * s)    
    Output = 1.0 / y
    if(Output>=0.5):
        return 1
    else:
        return 0  

def backpropagateEvaluator(y,no,j):
    global W1
    global W2
    global X
    global alpha
    global hidden_Layer_Nodes_output
    delta_error_output = -y/(no+0.0001) - (1-y)/(1-no+0.0001)
    
    for i in range(0,3):
        '''print a'''
        W2[i] = W2[i] - alpha*delta_error_output*no*(1-no)*hidden_Layer_Nodes_output[i]
    
    for m in range(0,2):
        for n in range(0,58):
                W1[m][n] = W1[m][n] - alpha*delta_error_output*hidden_Layer_Nodes_output[m]*(1-hidden_Layer_Nodes_output[m])*X[j][n]               
    

def feedforwardpredictor():
    global W1
    global W2
    global X
    global Train_actual_output
    global new_feature
    global hidden_Layer_Nodes_output
    
    for i in range(0,3000):    
        '''Node-1'''
        hidden_Layer_Nodes_output[0] = sigmoidal_function(W1[0],X[i])
        '''Node-2''' 
        hidden_Layer_Nodes_output[1] = sigmoidal_function(W1[1],X[i])
        
        new_feature[0] = hidden_Layer_Nodes_output[0]
        new_feature[1] = hidden_Layer_Nodes_output[1]
        
        thresholded_output = thresh_sigmoidal_function(W2,new_feature)
        normal_output = sigmoidal_function(W2,new_feature)
        
        if(thresholded_output != Train_actual_output[i]):
            backpropagateEvaluator(Train_actual_output[i],normal_output,i)
            


data = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 58))

Train_actual_output = [row[57] for row in data]

a = np.ones(3000)

hidden_Layer_Nodes_output = np.ones(3)

new_feature = np.ones(3)

Train_feature_vector = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 57))

X = np.column_stack((Train_feature_vector,a))

alpha = 0.1

W1 = np.random.rand(2,58)
'''print W1'''
W1[0][57] = 1
W1[1][57] = 1
W2 = np.random.rand(3)
W2[2] = 1 
p = np.ones(1600)


for i in range(0,15):
    feedforwardpredictor()



Test_feature_vector = np.loadtxt('TestX.csv', delimiter=',', usecols=range(0, 57))
b = np.ones(1600)
Test_feature_vector = np.column_stack((Test_feature_vector,b))
for i in range(0,1600):
    new_feature[0] = sigmoidal_function(W1[0],Test_feature_vector[i])
    new_feature[1] = sigmoidal_function(W1[1],Test_feature_vector[i])
    p[i] = thresh_sigmoidal_function(W2,new_feature)

with open('simple_Back_Propagation.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(0,1600):
        writer.writerow([i,int(p[i])])