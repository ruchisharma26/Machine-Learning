# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:58:02 2018

@author: Ruchi Awasthi
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Datasets
print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

l2_train_cll = np.zeros((10, 15))
l2_test_cll = np.zeros((10, 15))
l2_model_complexity = np.zeros((10,15))
l2_num_zero_weights  = np.zeros((10,15))
l1_num_zero_weights = np.zeros((10,15))

alphaList =  [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

#CWID Number: A20429225

for i in range(0,10):   
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=9225)
    for x in range(0, len(alphaList)):
        model =LogisticRegression(penalty='l2', C= alphaList[x], random_state=42)
        model.fit(X_train,y_train)   
        intercept = model.intercept_[0]
        coefficient = model.coef_[0]
        complexity = intercept*intercept + np.sum(coefficient**2)
     
        # L2 count zeros
        l2_count_zeros = len(coefficient)-np.count_nonzero(coefficient) + [1 if intercept==0 else 0][0]
        #CLL Calculations for train
        CLL_train =  model.predict_log_proba(X_train)
        sum_train =0 
        for j in range(0,len(y_train)):
            if y_train[j]== True:
                sum_train = sum_train + CLL_train[j][1]
            elif y_train[j]==False:
                sum_train = sum_train + CLL_train[j][0]
        l2_train_cll[i][x] = sum_train
        
        #CLL Calculations for test
        CLL_test = model.predict_log_proba(X_test)
        sum_test = 0
        for k in range(0,len(y_test)):
            if y_test[k]== True:
                sum_test= sum_test+ CLL_test[k][1]
            elif y_test[k]== False:
                sum_test = sum_test + CLL_test[k][0]     
        l2_test_cll[i][x] = sum_test
        
        
        l2_model_complexity[i][x] = complexity
        l2_num_zero_weights[i][x]  = l2_count_zeros
        
        #l1 LogisticRegression
        clf =LogisticRegression(penalty='l1', C= alphaList[x], random_state=42)
        clf.fit(X_train,y_train)   
        intercept_l1 = clf.intercept_[0]
        coefficient_l1 = clf.coef_[0]
        l1_count_zeros = len(coefficient_l1)-np.count_nonzero(coefficient_l1) + [1 if intercept_l1==0 else 0][0]
        l1_num_zero_weights[i][x]  = l1_count_zeros

#Graph plot for complexity and CLL

for i in range(0,10):  
    plt.plot(l2_model_complexity[i], l2_train_cll[i], color='g', label='Train Data')
    plt.plot(l2_model_complexity[i], l2_test_cll[i], color='orange', label='Test Data')
    plt.title('Dataset Number' + " " +str (i+1))
    plt.xlabel('model_complexity')
    plt.ylabel('CLL Values (Train and Test)')
    plt.legend()
    plt.show()
   
#Graph Plot for exponent and Number of Zeros
exponent_C =[-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
for i in range(0,10): 
    plt.plot(exponent_C, l2_num_zero_weights[i], color='g', label='L2 Zero Weights')
    plt.plot(exponent_C, l1_num_zero_weights[i], color='orange', label='L1 Zero Weights')
    plt.title('Dataset' + " " + str(i+1))
    plt.xlabel('Exponent of C')
    plt.ylabel('Number Of Zero Weights')
    plt.legend()
    plt.show()
    
np.set_printoptions(suppress=True)

print("L2 Model Complexity")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_model_complexity[i]))

print("\nCLL for Train")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_train_cll[i]))

print("\nCLL for Test")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_test_cll[i]))

print("\nL2 Zero Weights")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_num_zero_weights[i]))

print("\nL1 Zero Weights")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l1_num_zero_weights[i]))

# Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
pickle.dump((l2_model_complexity, l2_train_cll, l2_test_cll, l2_num_zero_weights, l1_num_zero_weights), open('result.pkl', 'wb'))
    
   
       
        
       


