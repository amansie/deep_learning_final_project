#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:42:06 2020

@author: kofiab
"""

import matplotlib.pyplot as plt
import numpy as np
'''
%matplotlib inline
%precision 16
'''

# PLOTTING
plt.rcParams['lines.linewidth'] = 3  
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}          
plt.rc('font', **font)

csfont = {'fontname':'Serif'}
plt.rcParams["font.family"] = "Arial"
plt.rcParams['figure.figsize'] = 6, 5
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# load data
#dnn_data = -np.log(np.loadtxt('dnn_mse_result_1.txt'))#dnn_mse_result_1.txt
#cnn_37x37_data= -np.log(np.loadtxt('37x37_pearson_mse_1.txt'))#37x37_pearson_mse_1.txt
#cnn_1d_data = -np.log(np.loadtxt('cnn.txt'))#-np.log(np.loadtxt('dylan-cnn_1-dim_2-layer_mse.txt'))#cnn.txt
linear_data = -np.log(np.loadtxt('linear_regression_results_ka2461_1.txt'))#linear_regression_results_ka2461_1.txt
gcn_data = -np.log(np.loadtxt('gcn_mse.txt'))








# plot
labels = ['Apoe', 'App', 'Gusb', 'Lamp5', 'Mbp', 'Pvalb', 'S100b', 'Slc30a3', 'Snca']
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(15, 10))
#rects1 = ax.bar(x - width*1.5, cnn_1d_data, width, label='CNN_1D')
#rects2 = ax.bar(x - width/2, cnn_37x37_data, width, label='VGG16')
rects3 = ax.bar(x - width/2, linear_data, width, label='LINEAR')
#rects4 = ax.bar(x + width*1.5, dnn_data, width, label='DNN')
rects5 = ax.bar(x + width/2, gcn_data, width, label='GCN')

ax.set_ylabel('-log(MSE)')
ax.set_title('MSE By Different Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('model_compare_log_mse.jpg')
plt.show()


####################showing percent error
# load data
dnn_data = (np.loadtxt('dnn_mse_result_percent_error.txt'))#dnn_mse_result_1.txt
#cnn_37x37_data=(np.loadtxt('37x37_pearson_mse_percent_error.txt'))#37x37_pearson_mse_1.txt
#cnn_1d_data = (np.loadtxt('cnn_percent_error.txt'))#-np.log(np.loadtxt('dylan-cnn_1-dim_2-layer_mse.txt'))#cnn.txt
linear_data = (np.loadtxt('linear_regression_results_ka2461_percent_error.txt'))#linear_regression_results_ka2461_1.txt
#gcn_data = (np.loadtxt('gcn_mse_percent_error.txt'))
gcn_2_data =  (np.loadtxt('gcn_second_percent_error.txt'))

fig, ax = plt.subplots(figsize=(15, 10))
'''
rects1 = ax.bar(x - width*1.5, cnn_1d_data, width, label='CNN_1D')
rects2 = ax.bar(x - width/2, cnn_37x37_data, width, label='VGG16')
rects3 = ax.bar(x + width/2, linear_data, width, label='LINEAR')
rects4 = ax.bar(x + width*1.5, dnn_data, width, label='DNN')
rects5 = ax.bar(x + width*2.5, gcn_data, width, label='GCN')
'''


rects1 = ax.bar(x - width/2, linear_data, width, label='LINEAR')
rects2 = ax.bar(x + width/2, gcn_2_data, width, label='GCN')

ax.set_ylabel('percent error')
ax.set_title('Percent error By Different Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('model_compare_percent_error.jpg')
plt.show()




####################showing z-squared
# load data
dnn_data =(np.loadtxt('dnn_mse_result_z-squared.txt'))#dnn_mse_result_1.txt
cnn_37x37_data= (np.loadtxt('37x37_pearson_mse_z-squared.txt'))#37x37_pearson_mse_1.txt
cnn_1d_data = (np.loadtxt('cnn_z-squared.txt'))#-np.log(np.loadtxt('dylan-cnn_1-dim_2-layer_mse.txt'))#cnn.txt
linear_data = (np.loadtxt('linear_regression_results_ka2461_z-squared.txt'))#linear_regression_results_ka2461_1.txt
gcn_data = (np.loadtxt('gcn_mse_z-squared.txt'))
gcn_2_data =  (np.loadtxt('gcn_second_z-squared.txt'))

'''
fig, ax = plt.subplots(figsize=(15, 10))
rects1 = ax.bar(x - width*1.5, cnn_1d_data, width, label='CNN_1D')
rects2 = ax.bar(x - width/2, cnn_37x37_data, width, label='VGG16')
rects3 = ax.bar(x + width/2, linear_data, width, label='LINEAR')
rects4 = ax.bar(x + width*1.5, dnn_data, width, label='DNN')
rects5 = ax.bar(x + width*2.5, gcn_data, width, label='GCN')
'''


fig, ax = plt.subplots(figsize=(15, 10))
#rects1 = ax.bar(x - width*1.5, cnn_1d_data, width, label='CNN_1D')
rects1 = ax.bar(x - width/2, linear_data, width, label='LINEAR')
rects2 = ax.bar(x + width/2, gcn_2_data, width, label='GCN')
#rects4 = ax.bar(x + width*1.5, dnn_data, width, label='DNN')
#rects5 = ax.bar(x + width*2.5, gcn_data, width, label='GCN')



ax.set_ylabel('z-score')
ax.set_title('z-score By Different Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('model_compare_z-squared.jpg')
plt.show()
