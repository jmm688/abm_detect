#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:59:33 2021

@author: jose
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import sklearn
import rasterio as rio
import seaborn as sns
import os





def get_array(path):

    rast = rio.open(path)
    image = rast.read()
    rast.close()
    
    return image

  
def plot_arr(array,color,arr_type): #############
  
  def plot_reg(arr, color):
    sns.heatmap(array[:,:], cmap = color)
    plt.title('Array_'+str(array.shape[0]), fontsize = 20)
    plt.show()
  
  
  def pc_plotter(array,color,pc):
  # array must be a 3-D array of dim [2,:,:]
  # color must be a string of accepted seaborn cmap colors 'gray', 'turbo', 'seismic', etc.
  # must be integer 1-3 to call which PC to plot, or string 'all' for PC1-3 plots.
  
    if pc == 'all':
        # Heatmap of PC-1
        sns.heatmap(array[0,:,:], cmap = color)
        plt.title('PC-1', fontsize = 20)
        plt.show()

        # Heatmap of PC-2
        sns.heatmap(array[1,:,:], cmap = color)
        plt.title('PC-2', fontsize = 20)
        plt.show()

        # Heatmap of PC-3
        sns.heatmap(array[2,:,:], cmap = color)
        plt.title('PC-3', fontsize = 20)
        plt.show()
        
    else:
        sns.heatmap(array[int(pc[-1]-1),:,:], cmap = color)
        plt.title('PC'+ str(pc), fontsize = 20)
        plt.show()
        
  
  if arr_type[0:2] == 'PC':
    pc_plotter(array,color,arr_type)
    
  elif arr_type == 'reg':
    plot_reg(array,color)
    
  else:
    print(arr_type,'WUUUT NOOOO!')
      


def get_precision(path):
  total_count = 0
  for i in os.listdir(path):
    
      # since a csv file with all scores per iteration exists, we parse it out
      if i[0:9] == 'partition':
          # print(i)
          file_path = path + '/' + i
          file_name = i  # for splitting and bool check!
          # extra check to weed out partition_num.tif or .xml etc...
          if os.path.isfile(file_path) and len(file_name.split('.')) == 2 and file_name.split('.')[1] == 'csv':
            partition = pd.read_csv(file_path, header=0, index_col=0)[0:367]

            # converting all floats as int and setting location of pixel as index
            df = partition.astype(int)
            df = df.set_index('location')
            
            for i in df.iterrows():
                if df.loc[i[0]]['prediction'] == df.loc[i[0]]['label']:
                    total_count += 1

                # else:
                #     template[i[0]] = 0
            
            #count += 1
           
  return total_count
  




gtif_path ='/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace/SVC/gtifs/ALL_partitions.tif'
#all_parts = get_array(gtif_path)



##########
### GETTING PRECISION

work_space_dir = '/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace'
#model_name = str(input('Enter model name here:  '))
model_name = '/SVC'
part_dir = work_space_dir + model_name





##########