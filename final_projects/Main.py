
"""
# Introduction to Gender Voice Recognation with Logistic Regression

# Index of Contents

* [Read Data and Check Features](#1)
* [Adjustment of Label values (male = 1, female = 0)](#2)
* [Data Normalization](#3)
* [Split Operation for Train and Test Data](#4)
* [Matrix creation function for initial weight values](#5)
* [Sigmoid function declaration](#6)
* [Forward and Backward Propogation](#7)
* [Updating Parameters](#8)
* [Prediction with Test Data](#9)
* [Logistic Regression Implementation](#10)
* [Logistic Regression with sklearn](#11)
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
# print(os.listdir("input"))

"""
Read Data and Check Features
"""

data = pd.read_csv("voice.csv")

# Get some information about our data
data.info()

"""
***Adjustment of Label values (male = 1, female = 0***
* After getting information about data we'll call male as 1 and female as 0***
"""

data.label = [1 if each == "male" else 0 for each in data.label]

data.info() # now we have label as integer

"""
***Data Normalization***
"""

y = data.label.values # main results male or female
x_data = data.drop(["label"], axis = 1) # prediction components

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values # all data evaluated from 1 to 0

"""
***Split Operation for Train and Test Data***
* Data is splitted for training and testing operations. We'll have %20 of data for test and %80 of data for train after split operation.
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Data Shapes
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

# Transform features to rows (Transpose)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

"""
***Matrix creation function for initial weight values***
"""

def initializeWeightsAndBias(dimension): # according to our data dimension will be 20
    w = np.full((dimension, 1), 0.01) 
    b = 0.0
    return w,b

"""
***Sigmoid function declaration***
"""

def sigmoid(z):
    y_head = (1 / (1 + np.exp(-z)))
    return y_head

"""
***Forward and Backward Propogation***
* Get z values from sigmoid function and calculate loss and cost. 
"""

x_train.shape[1]

def forward_backward_propogation(w, b, x_train, y_train):
    
    #forward propogation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1] # x_train.shape[1] is for scaling
    
    #backward propogation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] # x_train.shape[1] is for scaling
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # x_train.shape[1] is for scaling
    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
    
    return cost, gradients

"""
***Updating parameters***
* Our purpose is find to optimum weight and bias values using derivative of these values.
"""

def update(w, b, x_train, y_train, learningRate, numberOfIteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(numberOfIteration):
        # make forward and backward propogation and find costs and gradients
        cost,gradients = forward_backward_propogation(w, b, x_train, y_train)
        cost_list.append(cost)
        #lets update
        w = w - learningRate * gradients["derivative_weight"]
        b = b - learningRate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) paramters weights and bias
    parameters = {"weight" : w, "bias" : b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation = 'vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

"""
***Prediction with Test Data***
* Prediction using test data which is splitted first.
"""

def predict(w,b, x_test):
    # x_test is an input for forward propogation
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is Male (y_head = 1)
    # if z is smaller than 0.5, our prediction is Female (y_head = 0)
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    return Y_prediction

"""
***Logistic Regression Implementation***
"""

def logistic_regression(x_train, y_train, x_test, y_test, learningRate, numberOfIterations):
    dimension = x_train.shape[0] # that is 20 (feature count of data)
    w,b = initializeWeightsAndBias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learningRate, numberOfIterations)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    print("test accuracy : {} %.".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

#Let's try our model and check costs and prediction results.
logistic_regression(x_train, y_train, x_test, y_test, learningRate = 1, numberOfIterations = 100)

logistic_regression(x_train, y_train, x_test, y_test, learningRate = 1, numberOfIterations = 1000)

"""As you see above, when the iteration is increased, accuracy increasing too.

***Logistic Regression with sklearn***
* Logistic Regression Classification can be done with sklearn library. All codes which are written above correspond to the codes below.
"""


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
logistic_results=[]
mlp_relu_results=[]
mlp_sigmoid_results=[]
mlp_tanh_results=[]

tedad_layemakhfi_aval=10
tedad_layemakhfi_dovom=20

def darsad_max_list1_be_list2(list1,list2):
    s1=0
    for i in range(len(list1)): #dar inja len(list1)=len(list2)=50
        if max(list1[i],list2[i])==list1[i]:
            s1=s1+1
    return s1/len(list1)
tedad_testha=50
for i in range(1,tedad_testha+1):

    lr = LogisticRegression()
    lr.fit(x_train.T, y_train.T)
    logistic_results.append(lr.score(x_test.T, y_test.T)) # man inja az logisticregressiion
    #khod python gereftam ke chon chizi nemidim kolan sabete va ba oon moghayese kardam vali
    #ba logisticregression khodemoon ham mishod moghayese kard ke goftam be nazaram ba in moghayese she behtare



    clf_relu = MLPClassifier(hidden_layer_sizes=(tedad_layemakhfi_aval,tedad_layemakhfi_dovom),activation='relu',solver='adam',max_iter=700)#soal gofte epoch ro 100 dar nazar begirim
    #vali ba 100 eror mide ke hamgera nemishe va bayad bishtar gereft baray hamin 700 gereftam vali age moshkeli ba oon eror nist mishe 100 ham gereft va natija ro did...
    clf_relu.fit(x_train.T,y_train.T)
    mlp_relu_results.append(clf_relu.score(x_test.T, y_test.T))

    clf_sigmoid = MLPClassifier(hidden_layer_sizes=(tedad_layemakhfi_aval,tedad_layemakhfi_dovom),activation='logistic',solver='adam',max_iter=700)
    clf_sigmoid.fit(x_train.T,y_train.T)
    mlp_sigmoid_results.append(clf_sigmoid.score(x_test.T,y_test.T))

    clf_tanh = MLPClassifier(hidden_layer_sizes=(tedad_layemakhfi_aval,tedad_layemakhfi_dovom),activation='tanh',solver='adam',max_iter=700)
    clf_tanh.fit(x_train.T,y_train.T)
    mlp_tanh_results.append(clf_tanh.score(x_test.T,y_test.T))
    print("marhale {} baray moghayese relu,tanh,sigmoid anjam shod".format(i))
print("relu: {}".format(mlp_relu_results))
print("sigmoid: {}".format(mlp_sigmoid_results))
print("tanh:{}".format(mlp_tanh_results))
print("LogisticRegression:{}".format(logistic_results))
print("----------------------")
print("relu dar {} az mavared behtar az tanh boode".format(darsad_max_list1_be_list2(mlp_relu_results,mlp_tanh_results)))
print("relu dar {} az mavared behtar az sigmoid boode".format(darsad_max_list1_be_list2(mlp_relu_results,mlp_sigmoid_results)))
print("tanh dar {} az mavared behtar az sigmoid boode".format(darsad_max_list1_be_list2(mlp_tanh_results,mlp_sigmoid_results)))
print("----------------------")

x_plot=np.arange(1,tedad_testha+1,1)
plt.plot(x_plot,mlp_relu_results,'r-', label='relu')
plt.plot(x_plot,mlp_sigmoid_results,'b-',label='sigmoid')
plt.plot(x_plot,mlp_tanh_results,'g-',label='tanh')
plt.plot(x_plot,logistic_results,'y-',label='Logistic')
plt.xlabel("Number of test")
plt.ylabel("accuracy")
plt.legend()
plt.show()

mlp_tanh_results=[]
max_tedad_noronha=100
for i in range(1,max_tedad_noronha+1):
    #chon tanh amalkard behtari dar marhale ghabl neshoon dad kolan az tanh estefade mikonam
    clf_tanh = MLPClassifier(hidden_layer_sizes=(tedad_layemakhfi_aval, i), activation='tanh',
                             solver='adam', max_iter=700)
    clf_tanh.fit(x_train.T, y_train.T)
    mlp_tanh_results.append(clf_tanh.score(x_test.T, y_test.T))
    print("marhale {} baray moghayese tedad noron ha anjam shod".format(i))
x_plot=np.arange(1,max_tedad_noronha+1,1)
plt.plot(x_plot,mlp_tanh_results,'b-')
plt.xlabel("Number of neurons")
plt.ylabel("accuracy")
plt.show()

# baray moghayese drop out man chizi dar sklearn nadidam ke dropout ro hessab kone va man
# dar import kardan keras moshkel dashtam chon baray import kardan keras niaz be tensorflow
# bood va baray tensorflow niaz be nvidia toolkit/CUDA noskhe 11.2 bood ke in noskhe baray windows 11
# mojood nist...man ba site https://colab.research.google.com/#scrollTo=ufxBm1yRnruN run gereftam az in ghesmat code:

# code ba keras:
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#model = Sequential()
#tedad_layemakhfi_aval=10
#tedad_layemakhfi_dovom=20
#all_accuracy=[]
#for dropout in np.arange(0,1,0.1):# chon tanh behtar bood az natayej ghabl az tanh estefade kardam
#   model.add(Dropout(dropout))
#   model.add(Dense(tedad_layemakhfi_aval, input_dim=20, activation='tanh'))
#   model.add(Dense(tedad_layemakhfi_dovom, activation='tanh'))
#   model.add(Dense(1, activation='sigmoid')) # age bekhaim mishe nevesht:model.add(Dense(2, activation='softmax'))
#   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#   model.fit(x_train.T,y_train.T , epochs=100,batch_size=10)
#   _, accuracy = model.evaluate(x_test.T,y_test.T)
#   all_accuracy.append(accuracy)
#x_plot=np.arange(0,1,0.1)
#plt.plot(x_plot,all_accuracy,'r-')
#plt.xlabel("dropout")
#plt.ylabel("accuracy")
#plt.show()
