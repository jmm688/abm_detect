#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:12:54 2021

@author: jmm688
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.decomposition import PCA


import os
os.chdir('/home/jmm688/Tensorflow/TM_cnn_project/python_scripts')

# First we find the directory where all the hillshade data is stored
hs_path = '/home/jmm688/Tensorflow/TM_cnn_project/workspace/GIS_data/hillshades'  # -------- > look for the path and hardcode it!!!

if os.path.isdir(hs_path):
  os.chdir(hs_path)
  
else:
  print('These are not the folders you are looking for.    ---------- >  ',hs_path)
  print()
  print('*procedes to use the force...')
  print()
  print('...')
  print('..')
  print('.')
  print('.')
  print('.')


np_list = []         # making list of all hillshade numpy arrays. 
for i in os.listdir():
  hs = rio.open(i)
  data_ = hs.read()
  data = np.moveaxis(data_,0,-1)
  data = data.reshape([2000, 2000])  # ------------------------------- > INPUT HS DIMENSIONS!!!!
  np_list.append(data)

bool_check = True    # checking if all arrays are the same shape to continue
for i in np_list:
  for j in np_list:
    if i.shape == j.shape:
      pass
    
    else:
      print('dimensions of these arrays are not the same!')
      print('try again')
      print(i.shape)
    

# if all is good we now make TENSOR!!! woohoo
np_arr = np.zeros([len(np_list),j.shape[0],j.shape[1],1])



rgb = np.dstack((np_list[0].flatten(),np_list[1].flatten(),np_list[2].flatten(),np_list[3].flatten()))



#pca = PCA(n_components = 3)

#reduced_pca_data = pca.fit_transform(rgb[0,:,:])

#reduced_data = reduced_pca_data.reshape([2000,2000,3])





##################################################################
##########   VISUALIZE AND SAVE PC'S AS PNG IMAGES
##################################################################

execute_image = False

if execute_image:
  
  ### 1
  
  fig, ax = plt.subplots(figsize=(10,10))
  ax = sns.heatmap(reduced_data[:,:,0],ax=ax)
  fig.savefig('PC1.png',dpi=700)
  plt.show()
  
  
  ### 2
  
  fig, ax = plt.subplots(figsize=(10,10))
  ax = sns.heatmap(reduced_data[:,:,1],ax=ax)
  fig.savefig('PC2.png',dpi=700)
  plt.show()
  
  
  
  ### 3
  
  fig, ax = plt.subplots(figsize=(10,10))
  ax = sns.heatmap(reduced_data[:,:,2],ax=ax)
  fig.savefig('PC3.png',dpi=700)
  plt.show()
  
else:
  pass
###################################################################
###################################################################





###################################################################
#########  RESTORING AND STACKING AS AN RGB IMAGE TO SEND AS GTIF
###################################################################


# reduced_data IS the RGB :)    (2000, 2000, 3)
export_data = np.moveaxis(reduced_data,-1,0)

wkt_hs = hs.crs.to_wkt() 

# getting the transformation for this region
transform = hs.transform

new_dataset = rio.open('PCA_HS.tif', 'w', driver='GTiff',
                                          height = hs.height, 
                                          width = hs.width,
                                          count=reduced_data.shape[-1], 
                                          dtype=str(reduced_data.dtype),
                                          crs= wkt_hs,
                                          transform=transform)



new_dataset.write(export_data)
new_dataset.close()



























