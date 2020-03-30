#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:39:57 2020

@author: kofiab
"""

import pandas as pd  

dataset1 = pd.read_csv('/home/jupyter-kofi/new_data/output_train-1.csv')
dataset2 = pd.read_csv('/home/jupyter-kofi/new_data/output_test-1.csv')
output_dataset= dataset1.append(dataset2)
output_dataset.describe()
