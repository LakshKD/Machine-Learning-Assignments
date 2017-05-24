# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:23:05 2016

@author: laksh
"""

import numpy as np
import csv
import math as mp


def backpropagateEvaluator(y,f):
    global W1
    global W2
    global X
    global alpha
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    global final_output
    global Num_of_Output_Nodes
    
    del_E_by_delsigma_output = np.zeros(Num_of_Output_Nodes)
    del_E_by_delsigma_hidden = np.zeros(Num_of_Hidden_Nodes)
    one_hot = np.zeros(Num_of_Output_Nodes)
        
    one_hot[y] = 1
    
    #print one_hot
    for i0 in range(0,Num_of_Output_Nodes):
        a = -1*one_hot[i0]/(final_output[i0]+0.001)
        b = (1-one_hot[i0])/(1-final_output[i0]+0.001)
        del_E_by_delsigma_output[i0] = a - b
   
        
    #Updation Rule for Hidden Layer
    for m1 in range(0,Num_of_Hidden_Nodes):
        for j in range(0,Num_of_Output_Nodes):
            d = final_output[j]*(1-final_output[j])
            e = del_E_by_delsigma_output[j]*d
            del_E_by_delsigma_hidden[m1] = del_E_by_delsigma_hidden[m1]+e*W2[j][m1] 
   
    
    for j1 in range(0,Num_of_Hidden_Nodes):
         for n1 in range(0,58):
             k = hidden_Layer_Nodes_output[j1]*(1-hidden_Layer_Nodes_output[j1])
             g = del_E_by_delsigma_hidden[j1]*k
             W1[j1][n1] = W1[j1][n1] - alpha*g*X[f][n1]               
    
    
    #Updation Rule for Output Layer
    for i2 in range(0,Num_of_Output_Nodes):
        for j2 in range(0,Num_of_Hidden_Nodes):
            p = final_output[i2]*(1-final_output[i2])
            q = del_E_by_delsigma_output[i2]*p
            W2[i2][j2] = W2[i2][j2] - alpha*q*hidden_Layer_Nodes_output[j2]
    


def feedforwardpredictor():
    global W1
    global W2
    global X
    global Train_actual_output
    global hidden_Layer_Nodes_output
    global Num_of_Hidden_Nodes
    global final_output
    global del_E_by_delsigma_hidden
    #print Train_actual_output
    
    for i in range(0,3000):    
        
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_Layer_Nodes_output[j] =1/(1+mp.exp(-1*np.dot(W1[j],X[i])))
        #print hidden_Layer_Nodes_output    
        for k in range(0,Num_of_Output_Nodes):
             final_output[k] = 1/(1+mp.exp(-1*np.dot(W2[k],hidden_Layer_Nodes_output)))
        #print final_output
        thresholded_output = final_output.argmax()
        #print thresholded_output
        if(thresholded_output != Train_actual_output[i]):
            backpropagateEvaluator(Train_actual_output[i],i)

def Test_Data_Predictor():
    global p
    global Test_feature_vector
    hidden_output = np.zeros(Num_of_Hidden_Nodes)
    b = np.ones(1600)
    Test_feature_vector = np.loadtxt('TestX.csv', delimiter=',', usecols=range(0, 57))
    Test_feature_vector = np.column_stack((Test_feature_vector,b))
    
    for l in range(0,58):
        mi = np.min(Test_feature_vector[:,l])
        ma = np.max(Test_feature_vector[:,l])
        if (ma-mi)!=0:
            for k in range(0,1600):
                Test_feature_vector[k][l] = (Test_feature_vector[k][l]-mi)/(ma-mi)            
    
    for i in range(0,1600):
        for j in range(0,Num_of_Hidden_Nodes):
            hidden_output[j] = 1/(1+mp.exp(-1*np.dot(W1[j],Test_feature_vector[i])))
        for q in range(0,Num_of_Output_Nodes):
            p[i][q] = 1/(1+mp.exp(-1*np.dot(W2[q],hidden_output)))
            


data = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 58))

Train_actual_output = [row[57] for row in data]

a = np.ones(3000)

Train_feature_vector = np.loadtxt('Train.csv', delimiter=',', usecols=range(0, 57))

X = np.column_stack((Train_feature_vector,a))

#Normalizing X(feature) vector

for l in range(0,58):
    mi = np.min(X[:,l])
    ma = np.max(X[:,l])
    if (ma-mi)!=0:
        for k in range(0,3000):
            X[k][l] = (X[k][l]-mi)/(ma-mi)


Num_of_Hidden_Nodes = 25 #Number of hidden layer nodes

Num_of_Output_Nodes = 2  #Number of nodes at the Output Layer

hidden_Layer_Nodes_output = np.zeros(Num_of_Hidden_Nodes)

p = np.zeros((1600,2))

final_output = np.zeros(Num_of_Output_Nodes)

alpha = 0.08

W1 = np.ones((Num_of_Hidden_Nodes,58))
W2 = np.ones((Num_of_Output_Nodes,Num_of_Hidden_Nodes))
#W1 = np.random.rand(Num_of_Hidden_Nodes,58)
#W2 = np.random.rand(Num_of_Output_Nodes,Num_of_Hidden_Nodes)

"""
for weight1 in range(0,Num_of_Hidden_Nodes):
    for attribute1 in range(0,58):
        if W1[weight1][attribute1] < 0.5:
            W1[weight1][attribute1] = -1*W1[weight1][attribute1] 

for h_nodes in range(0,Num_of_Hidden_Nodes):
    W1[h_nodes][57] = 1


for weight2 in range(0,Num_of_Output_Nodes):
    for attribute2 in range(0,Num_of_Hidden_Nodes):
        if W2[weight2][attribute2] < 0.5:
            W2[weight2][attribute2] = -1*W2[weight2][attribute2] 

#print W1
#print W2
W1 = W1*0.0005
W2 = W2*0.0005
#print W1
#print W2
"""
for i in range(0,50):
    print i
    feedforwardpredictor()



Test_Data_Predictor()

for i in range(0,1600):
    print p[i].argmax()


"""
with open('Generalized_Back_Propagation_NN.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(0,1600):
        writer.writerow([i,p[i].argmax()])
"""