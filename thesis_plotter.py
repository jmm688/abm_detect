#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 20:33:57 2021

@author: jose


Just a quick script to figure out how to plot partition_csv

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import sklearn
import rasterio as rio

import os



# define location of partion file
partition_path = '/home/jose/Documents/Thesis/modeling/semantic_seg/SVC/SVC_test/partition_1.csv'

# loading in csv as numpy and using column 0 in csv as the index
partition = pd.read_csv(partition_path,header=0,index_col=0)

# converting all floats as int and setting location of pixel as index
df = partition.astype(int)
df = df.set_index('location')

# creating a template of NAN values of original image dimension ### FIXME: AUTOMATE!!!
template = np.zeros([400,400]).flatten() 
template[:] = np.nan
 

'''
# tuple operation where i[0] is the index number integer and i[1] is pd.series
so i[1][0] is the prediction column and i[1][1] is label   # FIXME: comment needs revision

'''

for i in df.iterrows():
  if df.loc[i[0]]['prediction'] == df.loc[i[0]]['label']:
    template[i[0]] = 1
    
  else:
    template[i[0]] = 0
    
    
final = template.reshape([400,400])    

  
