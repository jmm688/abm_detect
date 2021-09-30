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

wv3_path = '/home/jose/Downloads/fixed_field_area_modified_TEST.tif'

work_space_dir = '/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace'
#model_name = str(input('Enter model name here:  '))
model_name = '/SVC'
part_dir = work_space_dir + model_name




"""
This function will produce template arrays from partitions for plotting or gtif puposes
The function get_plot will take 2 arguments...
1) path: The the path of a csv file containing model results
2) plot_type: what kind of dimension to expect, so template creation can be adequate 
  - argument can either be 'partition' which will return an array of dimensions [1,xdim,ydim] or
    or 'total' which will return an array of dimensions [num_partitions, xdim, ydim]
"""
def get_plot(path):

    # # the heavy lifting of formating array into image-like dimensions
    def p_iterator(df):  # iterates over a pandas dataframe to produce a template in image formate for plotting in sns or send to gtif

        # creating a template of NAN values of original image dimension ### FIXME: AUTOMATE!!!
        template = np.zeros([400, 400]).flatten()
        template[:] = np.nan

        # will look through each row and evaluate if the prediction matched label
        for i in df.iterrows():
            if df.loc[i[0]]['prediction'] == df.loc[i[0]]['label']:
                template[i[0]] = 1

            else:
                template[i[0]] = 0

        return template

# ------------------------------------------------------------------------------
#     WHEN PATH IS AN INDIVIDUAL FILE:  PLOT WILL BE PER PARTITION
# ------------------------------------------------------------------------------

    # When path is a file! (individual model partition plots!)
    if os.path.isfile(path):
        partition = pd.read_csv(path, header=0, index_col=0)

        # converting all floats as int and setting location of pixel as index
        df = partition.astype(int)
        df = df.set_index('location')
        template = p_iterator(df)

        final = template.reshape([1, 400, 400])
        return final

# ------------------------------------------------------------------------------
#   WHEN PATH IS A DIRECTORY:  PLOT WILL BE ALL PARTITION IN ONE ARRAY
# ------------------------------------------------------------------------------
#

    elif os.path.isdir(path):
        # np.load(file_path).shape[-1] maybe?? (86)
        super_template = np.zeros([86, 400, 400])
        super_template[:, :, :] = np.nan

        #print('holy!')
        count = 0
        list_array = []
        for i in os.listdir(path):
            # since a csv file with all scores per iteration exists, we parse it out
            if i[0:9] == 'partition':
                # print(i)
                file_path = part_dir + '/' + i
                file_name = i  # for splitting and bool check!
                # extra check to weed out partition_num.tif or .xml etc...
                if os.path.isfile(file_path) and len(file_name.split('.')) == 2 and file_name.split('.')[1] == 'csv':
                    # count+=1
                    # print(type(file_name.split('.')[1]))

                    # re passing into this function but as a single file now
                    arr = get_plot(file_path)
                    # sns.heatmap(arr[0,:,:])
                    # plt.show()
                    super_template[count, :, :] = arr[0, :, :]
                    # count+=1
                    # print(file_name, count)
                    # print()
                    count += 1
                    # break
                    # get_gtif(write_dir + '/' + file_name.split('.')[0], arr, wv3_path)  # RELIC!!! for gtif!!!! keep uncomented!!!

    #        return super_template

                else:
                    return 'THIS PATH IS NOT A FILE or not a csv!!---->  ' +  str(file_path)

            # KEEP as PASS! might need a boolean check late!
            else:
                pass

        return super_template

def get_gtif(export_path, array, image_reference_path, name):
    
    folder_name = export_path + '/gtifs'
    if os.path.isdir(folder_name):
      write_dir = folder_name

    else:
      os.mkdir(folder_name)
      write_dir = folder_name
    
    
    ref = rio.open(image_reference_path)
    wkt_ref = ref.crs.to_wkt()
    crs = ref.crs

    # getting the transformation for this region
    transform = ref.transform

    new_dataset = rio.open(folder_name + name + '.tif', 'w', driver='GTiff',
                           height=ref.height,
                           width=ref.width,
                           count=array.shape[0],
                           nodata=np.nan,
                           dtype=str(array.dtype),
                           crs=wkt_ref,
                           transform=transform)

    new_dataset.write(array)





'''
This function will execute all functions! it will do this in order to get all 
partition data into array [num_partitions, xdim, ydim] using function get_plot.

From here it will then send to get_gtif function to generate 3 geotifs: All_partitions[num_partitions, xdim, ydim],
all_sum[1, xdim, ydim] and average[1, xdim, ydim].

Finally, it will return a tuple object named partition_stats where:
  - partition_stats[0] is the array ---> all_sum[1, xdim, ydim]
  - partition_stats[1] is the array ---> average[1, xdim, ydim]
  
  
Input:
  
1)path_to_partition_directory: A file path to the location where ALL partition csv files exist

2)image_reference_path: A file path to the location of geotif(raster) that partition modeling was based on
  - This is needed to write new gtifs seing as GDAL requires a wkt string and transformation to locate where each pixel plots on the planet!

3) ref_path: a file path to any partition.csv file so that location of each mine pixel can be located in order to sum and average
  - This is because partition_num.csv generated by software contain the exact same label = 1 test pixels for all iterations.
'''
def get_gtif_mines(path_to_partition_directory, image_reference_path, ref_path):   # takes all csv in a directory and will stack all mine related pixels into one dataset to be exported

  
  # extra function to help with creating templates for input
  def average(array):  # should return an array of dimensions [1,xdim, ydim]
    sum_template = np.zeros([1,array.shape[1],array.shape[2]])
    
    for i in range(0,array.shape[0]):
      sum_template = np.where(array[i,:,:] > 0, (sum_template + 1), array[i,:,:])
    
    avg_template = np.where(sum_template is not np.nan, sum_template / array.shape[0], sum_template)
      
      
    return sum_template, avg_template


  # Actual start of function :)
  if os.path.isfile(ref_path):
      ref_df = pd.read_csv(ref_path, header=0, index_col=0)

      # converting all floats as int and setting location of pixel as index
      rdf = ref_df.astype(int)
      rdf = rdf.set_index('location')
      
      ref_template = np.zeros([400, 400]).flatten()
      ref_template[:] = np.nan
      
      for i in rdf.iterrows():
          if rdf.loc[i[0]]['label'] == 1:
              ref_template[i[0]] = 1

          else:
              pass
   
      # a new array with the classification result at given known mine locations in ref_template data frame 
      arr = get_plot(part_dir)   #[86,400,400]  NOTE for me :)
      
      template = np.zeros([arr.shape[0],arr.shape[1],arr.shape[2]]) # CHHAHAHAHAANGE!!!!!!!!!!!!!!!
      template[:] = np.nan
      
      count = 0
      for i in range(0, arr.shape[0]):
        mines = np.where(ref_template == 1, arr[count,:,:].flatten(), ref_template)
        mines = mines.reshape([1,400,400])
        
        template[count,:,:] = mines[0,:,:]
        #
        count+=1
      
      #a
      partition_stats = average(template)
      sum_tif = get_gtif(path_to_partition_directory, partition_stats[0], image_reference_path, '/all_sum')
      average_tif = get_gtif(path_to_partition_directory, partition_stats[1], image_reference_path, '/average')
      ALL_parts_tif = get_gtif(path_to_partition_directory, arr, image_reference_path, '/ALL_partitions')
     
      return partition_stats



############################################################################################
### This will execute getting plots for all partitions and pass as argument to make gtif

#all_parts = get_plot(part_dir)
#get_gtif('All_partitionss', all_parts, wv3_path)  # ---------> define path to write to
#get_gtif_mines('/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace/SVC', all_parts, wv3_path,name)
############################################################################################


############################################################################################
### This will execute getting plots for all partitions and pass as argument to make gtif


part_dir = part_dir = work_space_dir + model_name
wv3_path = '/home/jose/Downloads/fixed_field_area_modified_TEST.tif'
part_ref = '/home/jose/Github/my_projs/abm_detect/modeling/thesisoftware_workspace/SVC/partition_1.csv' # helps function see where pixels = 1, and location (which are the mines!)
data = get_gtif_mines(part_dir, wv3_path, part_ref)
#################################################################################################################



