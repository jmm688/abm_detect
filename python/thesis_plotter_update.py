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
import seaborn as sns
import os



# define location of partion file
partition_path = '/home/jose/Documents/Thesis/modeling/semantic_seg/SVC/SVC_test/partition_1.csv'


wv3_path = '/home/jose/Downloads/fixed_field_area.tif'


work_space_dir = '/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace'
#model_name = str(input('Enter model name here:  '))
model_name = '/SVC'
part_dir = work_space_dir + model_name


# loading in csv as numpy and using column 0 in csv as the index

# function takes array and a reference image to extract necessary infomation for writting a Gtif using Rasterio + GDAL
def get_gtif(export_path, array, image_reference_path):
  ref = rio.open(image_reference_path)
  wkt_ref = ref.crs.to_wkt() 
  crs = ref.crs
  
  # getting the transformation for this region
  transform = ref.transform
  
  print(ref.shape)
  print(array.shape)
  
  print()
  print(wkt_ref)
  print(type(transform))
  print()
  print(ref.shape[-1])
  
  
  # new_dataset = rio.open(export_path, 'w', driver='GTiff',
  #                                           height = ref.height, 
  #                                           width = ref.width,
  #                                           count = 1,
  #                                           nodata = np.nan,
  #                                           dtype=str(arr.dtype),
  #                                           crs= wkt_ref,
  #                                           transform=transform)



  #new_dataset.write(array)
  
  
  #print(ref.count)
  ref.close
  
  return transform
  
  


def get_plot(path):
  partition = pd.read_csv(path,header=0,index_col=0)
  
  # converting all floats as int and setting location of pixel as index
  df = partition.astype(int)
  df = df.set_index('location')
  
  # creating a template of NAN values of original image dimension ### FIXME: AUTOMATE!!!
  template = np.zeros([400,400]).flatten() 
  template[:] = int(-999)


  '''
  # tuple operation where i[0] is the index number integer and i[1] is pd.series
  #so i[1][0] is the prediction column and i[1][1] is label   # FIXME: comment needs revision
  
  '''
  for i in df.iterrows():
    if df.loc[i[0]]['prediction'] == df.loc[i[0]]['label']:
      template[i[0]] = 1
      
    else:
      template[i[0]] = 0
    
  
  
  final = template.reshape([400,400])
  #sns.heatmap(final[0,:,:],cmap='seismic')
  #plt.show()
  
  return final





test = get_plot(partition_path)

for i in os.listdir(part_dir):
    if i[0:9] == 'partition':     # since a csv file with all scores per iteration exists, we parse it out
      #print(i)
      file_path = part_dir + '/' + i
      if os.path.isfile(file_path):
        arr = get_plot(file_path)
        tr= get_gtif(file_path, arr, wv3_path)
    
        break
        
      
        pass
    else:
        print(i)


