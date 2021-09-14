#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:24:03 2021

@author: jose
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



# function will take a color PCA plot and extract the single PC grayscale and write to a Gtiff
def PC_writer(pca_rgb):
  
  if os.path.isfile(pca_rgb):
    #os.chdir(hs_path)
    #print('Woot')
    
    # opening rgb raster as np.array
    rast = rio.open(pca_rgb)
    data = rast.read()    # getting np.array
    #sns.heatmap(data[-1,:,:])    # defining we want the 3rd component
    
    
    # raster info to transform numpy array to geotiff
    wkt_hs = rast.crs.to_wkt()    # getting wkt string from rio object
    transform = rast.transform    # getting the transformation for this region from rio object
    
    # getting dimension info and closing rio object
    height = rast.height
    width = rast.width
    rast.close()  # closing rio object
    
    export_data = data[-1,:,:]
    export_data = export_data.reshape([1,export_data.shape[0],export_data.shape[1]])
    print(export_data.shape)
    #print(data.shape)
    
    
     # writing gtiff for GMT  *** FIXME: use with rasterio so open twice doesnt happen!!! ***
    new_dataset = rio.open('test.tif', 'w', driver='GTiff',
                                               height = height, 
                                               width = width,
                                               count=export_data.shape[0], 
                                               dtype=str(data.dtype),
                                               crs= wkt_hs,
                                               transform=transform)
    
    
    new_dataset.write(export_data)
    new_dataset.close
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
butt = PC_writer('/home/jose/Documents/Thesis/Figures/GMT/PCA/PCA_HS.tif')
  
