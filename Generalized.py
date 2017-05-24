# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:01:02 2016

@author: laksh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 06:06:28 2016

@author: laksh
"""
import numpy as np
import csv
import math as mp
def sigmoidal_function(w,x):
    #print w
    s = np.dot(w,x)
    #print s
    if s < -100:
        return 0
    elif(s > 100):
        return 1
    y = 1.0 + m.exp(-1.0 * s)
         
    Output = 1.0 / y
    #print y
    return Output
        
def backpropagateEvaluator(y,f,del_E_by_delsigma_hidden):
    global W1
    global W2
    global X
    global alpha
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    global normal_output
    global Num_of_Output_Nodes
    #global del_E_by_delsigma_hidden
    
    
    #print f 
    one_hot = np.zeros(Num_of_Output_Nodes)
        
    one_hot[y] = 1
    #print y
    print one_hot
    #print normal_output
    for i0 in range(0,Num_of_Output_Nodes):
        delta_error_output[i0] = -1*one_hot[i0]/(normal_output[i0]+0.001)-(1-one_hot[i0])/(1-normal_output[i0]+0.001)
    #print delta_error_output
        
    #Updation Rule for Hidden Layer
    for m1 in range(0,Num_of_Hidden_Nodes):
        for j in range(0,Num_of_Output_Nodes):
            del_E_by_delsigma_hidden[m1] = del_E_by_delsigma_hidden[m1]+delta_error_output[j]*normal_output[j]*(1-normal_output[j])*W2[j][m1] 
    #print del_E_by_delsigma_hidden
    
    for j1 in range(0,Num_of_Hidden_Nodes):
         for n1 in range(0,58):
                W1[j1][n1] = W1[j1][n1] + alpha*del_E_by_delsigma_hidden[j1]*hidden_Layer_Nodes_output[j1]*(1-hidden_Layer_Nodes_output[j1])*X[f][n1]               
    #print "Updated weight W1",W1
    
    #Updation Rule for Output Layer
    for i2 in range(0,Num_of_Output_Nodes):
        for j2 in range(0,Num_of_Hidden_Nodes):
            W2[i2][j2] = W2[i2][j2] + alpha*delta_error_output[i2]*normal_output[i2]*(1-normal_output[i2])*hidden_Layer_Nodes_output[j2]
    #print "Updated weight W2",W2

def Test_Data_Predictor():
    global hidden_output   
    global p
    global Test_feature_vector
    
    b = np.ones(1600)
    Test_feature_vector = np.column_stack((Test_feature_vector,b))
    
    for l in range(0,58):
        #mi = np.min(Test_feature_vector[:,l])
        ma = np.max(Test_feature_vector[:,l])
        if (ma)!=0:
            for k in range(0,1600):
                Test_feature_vector[k][l] = (Test_feature_vector[k][l])/(ma)
                
    
    for i in range(0,1600):
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_output[j] = 1/(1+mp.exp(-1*np.dot(W1[j],Test_feature_vector[i])))
        for q in range(0,Num_of_Output_Nodes):
            p[i][q] = 1/(1+mp.exp(-1*np.dot(W2[q],hidden_output)))
            

def feedforwardpredictor():
    global W1
    global W2
    global X
    global Train_actual_output
    global new_feature
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    global normal_output
    global softmax_output
    global final_output
    global del_E_by_delsigma_hidden
    #print Train_actual_output
    
    for i in range(0,3000):    
        del_E_by_delsigma_hidden = np.zeros(Num_of_Hidden_Nodes)
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_Layer_Nodes_output[j] =1/(1+mp.exp(-1*np.dot(W1[j],X[i])))
        #print hidden_Layer_Nodes_output     
        #for j in range(0,Num_of_Hidden_Nodes):
            #new_feature[j] = hidden_Layer_Nodes_output[j]
        #print W2
        for j in range(0,Num_of_Output_Nodes):
             normal_output[j] = 1/(1+mp.exp(-1*np.dot(W2[j],hidden_Layer_Nodes_output)))
        #print normal_output
        thresholded_output = normal_output.argmax()
        #print thresholded_output
        if(thresholded_output != Train_actual_output[i]):
            #print Train_actual_output[i] 
            backpropagateEvaluator(Train_actual_output[i],i,del_E_by_delsigma_hidden)
            
        
data = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 58))

Train_actual_output = [row[57] for row in data]

a = np.ones(3000)

Train_feature_vector = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 57))

X = np.column_stack((Train_feature_vector,a))

Num_of_Hidden_Nodes = 57 #Number of hidden layer nodes

Num_of_Output_Nodes = 2  #Number of nodes at the Output Layer

delta_error_output = np.zeros(Num_of_Output_Nodes)

hidden_Layer_Nodes_output = np.zeros(Num_of_Hidden_Nodes)

del_E_by_delsigma_hidden = np.zeros(Num_of_Hidden_Nodes)

new_feature = np.zeros(Num_of_Hidden_Nodes)



hidden_output = np.zeros(Num_of_Hidden_Nodes) 

normal_output = np.zeros(Num_of_Output_Nodes)

softmax_output = np.zeros(Num_of_Output_Nodes)

final_output = np.zeros(Num_of_Output_Nodes)

Test_feature_vector = np.loadtxt('TestX.csv', delimiter=',', usecols=range(0, 57))

for l in range(0,58):
    #mi = np.min(X[:,l])
    ma = np.max(X[:,l])
    if (ma)!=0:
        for k in range(0,3000):
            X[k][l] = (X[k][l])/(ma)

alpha = 0.01

p = np.zeros((1600,2))
 
W1 = np.random.rand(Num_of_Hidden_Nodes,58)
for weight1 in range(0,Num_of_Hidden_Nodes):
    for attribute1 in range(0,58):
        if W1[weight1][attribute1] < 0.5:
            W1[weight1][attribute1] = -1*W1[weight1][attribute1] 

for h_nodes in range(0,Num_of_Hidden_Nodes):
    W1[h_nodes][57] = 1
    
W1 = W1*0.05


W2 = np.random.rand(Num_of_Output_Nodes,Num_of_Hidden_Nodes) 

for weight2 in range(0,Num_of_Output_Nodes):
    for attribute2 in range(0,Num_of_Hidden_Nodes):
        if W2[weight2][attribute2] < 0.5:
            W2[weight2][attribute2] = -1*W2[weight2][attribute2] 

W2 = W2*0.05

#print "Initial weight W1",W1
#print "Initial weight W2",W2


for i in range(0,10):
    print i
    feedforwardpredictor()

Test_Data_Predictor()

with open('Generalized_Back_Propagation_NN.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(0,1600):
        writer.writerow([i,p[i].argmax()])
