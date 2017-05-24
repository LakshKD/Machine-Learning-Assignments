# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 06:06:28 2016

@author: laksh
"""
import numpy as np
import csv
import math as m

def sigmoidal_function(w,x):
    s = np.dot(w,x)
    if s < -100:
        return 0
    elif(s > 100):
        return 1
    y = 1.0 + m.exp(-1.0 * s)    
    Output = 1.0 / y
    return Output
        
def thresh_sigmoidal_function(w,x):
    s = np.dot(w,x)
    if s < -100:
        return 0
    elif(s > 100):
        return 1
    y = 1.0 + m.exp(-1.0 * s)    
    Output = 1.0 / y
    '''print (Output)'''
    if(Output>0.5):
        return 1
    else:
        return 0  

def backpropagateEvaluator(y,no,f):
    global W1
    global W2
    global X
    global alpha
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    delta_error_output = -y/(no) - (1-y)/(1-no)
    
    for i in range(0,Num_of_Hidden_Nodes+1):
        '''print a'''
        W2[i] = W2[i] + alpha*delta_error_output*no*(1-no)*hidden_Layer_Nodes_output[i]
    
    for m1 in range(0,Num_of_Hidden_Nodes):
        for n in range(0,58):
                W1[m1][n] = W1[m1][n] + alpha*delta_error_output*hidden_Layer_Nodes_output[m1]*(1-hidden_Layer_Nodes_output[m1])*X[f][n]               



def Test_Data_Predictor():
    global hidden_output   
    global p
    global Test_feature_vector
    
    b = np.ones(1600)
    Test_feature_vector = np.column_stack((Test_feature_vector,b))
    
    for l in range(0,58):
        mi = np.min(Test_feature_vector[:,l])
        ma = np.max(Test_feature_vector[:,l])
        if (ma-mi)!=0:
            for k in range(0,1600):
                Test_feature_vector[k][l] = (Test_feature_vector[k][l] - mi)/(ma-mi)
                
    
    for i in range(0,1600):
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_output[j] = sigmoidal_function(W1[j],Test_feature_vector[i])
        p[i] = thresh_sigmoidal_function(W2,hidden_output)    

def feedforwardpredictor():
    global W1
    global W2
    global X
    global Train_actual_output
    global new_feature
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    
    for i in range(0,3000):    
        
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_Layer_Nodes_output[j] = sigmoidal_function(W1[j],X[i])
        
        for j in range(0,Num_of_Hidden_Nodes):
            new_feature[j] = hidden_Layer_Nodes_output[j]
        
        
        thresholded_output = thresh_sigmoidal_function(W2,new_feature)
        normal_output = sigmoidal_function(W2,new_feature)
        
        if(thresholded_output != Train_actual_output[i]):
            backpropagateEvaluator(Train_actual_output[i],normal_output,i)
            
        
data = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 58))

Train_actual_output = [row[57] for row in data]

a = np.ones(3000)

Num_of_Hidden_Nodes = 57

hidden_Layer_Nodes_output = np.ones(Num_of_Hidden_Nodes + 1)

new_feature = np.ones(Num_of_Hidden_Nodes + 1)

Train_feature_vector = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 57))

X = np.column_stack((Train_feature_vector,a))

hidden_output = np.ones(Num_of_Hidden_Nodes+1) 

Test_feature_vector = np.loadtxt('TestX.csv', delimiter=',', usecols=range(0, 57))

for l in range(0,58):
    mi = np.min(X[:,l])
    ma = np.max(X[:,l])
    if (ma-mi)!=0:
        for k in range(0,3000):
            X[k][l] = (X[k][l] - mi)/(ma-mi)

alpha = 0.5

p = np.zeros(1600)
 
W1 = np.random.rand(Num_of_Hidden_Nodes,58)*0.05
for j in range(0,Num_of_Hidden_Nodes):
    W1[j][57] = 0.05


W2 = np.random.rand(Num_of_Hidden_Nodes+1)
W2[Num_of_Hidden_Nodes] = 0.05 



for i in range(0,1):
    feedforwardpredictor()

Test_Data_Predictor()

with open('Back_Propagation_NN.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(0,1600):
        writer.writerow([i,int(p[i])])
