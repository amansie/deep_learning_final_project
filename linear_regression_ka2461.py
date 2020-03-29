#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:29:02 2020

@author: kofiab
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# given a gene run a regression and give the Mean Square Error
def operateRegression(gene):
    input_train_dataset = pd.read_csv('/home/jupyter-kofi/new_data/input_train.csv')
    output_train_dataset = pd.read_csv('/home/jupyter-kofi/new_data/output_train-1.csv')
    X_train=input_train_dataset.values
    Y_train=output_train_dataset[gene].values
    regressor = LinearRegression()  
    regressor.fit(X_train[:,1:],Y_train)
    input_test_dataset = pd.read_csv('/home/jupyter-kofi/new_data/input_test.csv')
    X_test=input_test_dataset.values
    y_pred = regressor.predict(X_test[:,1:])
    output_test = pd.read_csv('/home/jupyter-kofi/new_data/output_test-1.csv')
    y_test=output_test[gene].values
    #Check the difference between the actual value and predicted value.
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print('Mean Squared Error:'+gene+' : ', metrics.mean_squared_error(y_test, y_pred))  
    
    
operateRegression('App')
operateRegression('Apoe')
operateRegression('Gusb')
operateRegression('Lamp5')
operateRegression('Pvalb')
operateRegression('Rorb')
operateRegression('S100b')
operateRegression('Slc30a3')
operateRegression('Snca')
operateRegression('Mbp')
