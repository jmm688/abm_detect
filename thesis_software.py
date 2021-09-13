#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:45:21 2021

@author: josemarmolejo
"""



import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import sklearn

import os

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix


import rasterio as rio


np.set_printoptions(suppress=True)










##############################################################################    
##############################################################################
##############################################################################
##############################################################################    
#################                   CLASS                      ###############
##############################################################################
##############################################################################    
#################                  PREPROC                     ###############
##############################################################################
##############################################################################    
##############################################################################
##############################################################################

'''
The purpose of this class is to facilitate preprocessing steps required for other 
more complicated statistical models.

This includes:
  1) probing data for information such as raster(matrix) count/size, class
     partition operations etc.
  
  2) input important parameters such as x and y dimensions of a field area,
     band count or spectral band information.
     
  3) creating raster objects for both optical multispectral images and y label/
     variate images.
     
  4) And finally writes tensor objects to a numpy binary.npy file with unique 
     dimensional attributes needed for the statistical modeling python class
     later used in the program
         - this binary file includes both optical and label rasters in one :)
  
  
  
class functions:
  
  
  
flat_arr(self)
  
  Takes an array and reorders axes to fit GIS multispectral standard and flattens into a 1-dimensional. 
  
  examples:
    X.flatten_arr()
        or
    y.flatten_arr()
    






partition_log(X, y)
  
  X should be the data set
  y should be the label for X data
  
  Takes two raster (preferably raster objects) as arguments and checks if they
  are equal spatial dimensions. Then it will print both class information such
  as class count, and the needed partition count to equalize a biased binary 
  data set.
  
  
  examples:
    preproc.partition_log(X, y)
    
    





get_count(array, class_num)

  The purpose of this function is to pass an y like array or label array and
  return the class count or occurance. The class_num value must be an integer
  present in the data set.
  
  examples:
    preproc.get_count(array, class_num = 1)
      returns the times that class 1 occurs in the data set named array.
      
      
      
      
      
      
      
      
partition_np(X,y)

  If the class preproc were a cell.... this function would be the equivalent to
  the mitochondria!!!
  
  It takes two arrays of equal spatial dimensions and packages it into a nice 
  little tensor object to write as a binary.npy file
  This object will have information regarding the data within array X and its 
  label y. in addition it will have been shuffled and encoded with is spatial
  position in x and y cartesian space.
  
  Just run this command in python (working on running it as an executeble on the
  command line :) ) and BAM!!!! a magical file appears in the processing
  directory! woohoo!!! 
  
  examples:
    preproc.partition_np(X,y)
    X is the dataset
    y is the label dataset
  
     

'''


class preproc:
  
  
  # class variabels
  
  xdim = 400    # cropped out E - W direction 
  ydim = 400    # cropped out N - S direction
  wv3dim = 16   # band or spectral resolution 
  
  band = {
              1 : 'Coastal', 
              2 : 'Blue',
              3 : 'Green',
              4 : 'Yellow',
              5 : 'Red',
              6 : 'RedEdge',
              7 : 'NIR1',
              8 : 'NIR2',
              9 : 'SWIR-1',
              10: 'SWIR-2',
              11: 'SWIR-3',
              12: 'SWIR-4',
              13: 'SWIR-5',
              14: 'SWIR-6',
              15: 'SWIR-7',
              16: 'SWIR-8'
              
              }
  

  
  # train test split
  train = 0.2
  test =  0.8
  
  
  
  # loading in image
  def __init__(self, path):
    
    #creates a rasterio object 
    self.raster = rio.open(path, mode='r')
    
    # the raw numpy array without processing
    self.array = self.raster.read()[:,0:preproc.xdim,0:preproc.ydim]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##############################################################################    
#################        CLASS FUNCTIONS for preproc           ###############
##############################################################################
    

# ( 1 )
###    Flat array for more usfull stuff!   ###
  
  def flat_arr(self):
    
    # if it is multispectral image
    if self.array.shape[0] == preproc.wv3dim:
  
    # creates a numpy array that will be correctly processed
      flat = self.array
      
      # this is the flattened array 
      flat = np.moveaxis(flat, 0, -1)
      flat = flat.reshape(160000,16)
      
    # if it is the target or variate (answers)  
    else:
      
      flat = self.array
      
      # this is the flattened array 
      flat = np.moveaxis(flat, 0, -1)
      flat = flat.reshape(160000,1)
    
    
    
    return flat
    






# ( 2 )
###    Creating a usefull pandas df from a flat np array   ###

  def flat_df(self):
    
    # if multispectral
    if self.array.shape[0] == preproc.wv3dim:

      # writing list of band names to be passed into dataframe constructor
      bandcolumns = [preproc.band[band] for band in range(1,len(preproc.band)+1)]
  
      # writing pandas df of this flattened arr 
      df = pd.DataFrame(self.flat_arr(), columns=bandcolumns)
      
    # if target or variate  
    else:
      
      df = pd.DataFrame(self.flat_arr())
  
    
    
    return df
  
  
  
  
  
  
  
  
  
  # returns text info regarding partition stuff.
  def partition_log(X, y):
    message = ''
    X = X.flat_arr()
    y = y.flat_arr()
    
    
    # checking if dimensions match
    if X.shape[0] == y.shape[0]:
      countx = 0
      county = 0
      class_1 = []
      
      # counting time 
      for i in y:
        if i == 1:
          countx+=1
        
        # i left this in elif in case one day I decide to add multiclass and not just binary  
        elif i == 0:
          county +=1
          
      
      print()   
      print('='*45)
      print(' - - - - - - - ')
      print(' Partion log:')
      print(' - - - - - - - ')
      print()
      print('...')
      print('...')
      print('...')
      print()
      print()
      print()
      print('There are ' + str(countx) + ' mine pixels')
      print('-'*45)
      print()
      print('There are ' + str(county) + ' NON mining pixels')
      print('-'*45)
      print()
      print('This is a 1:' + str(county/countx)[0:4] + ' ratio!')
      print('-'*45)
      print()
      print('Class 1 only represents ' + str(countx/county * 100)[0:4] + '% of the data')
      print('-' *45)
      
      
      
      pass
      
      
      
    else: 
      message = 'Error... the pixel dimensions do not match!'
      print(message)
    
    #return butt
    pass





  
  
  # will return the count of pixels in a certain class
  # MUST be in shape (:,)!!!! it should only take in label array.
  # takes as arguments: 
      # A. The array object
      # B. The numerical class to be counted 
  def get_count(array, class_num):
    
    array = array.flat_arr()
    
    count = 0
    
    if class_num == 0:
      
      for i in array:
        if i == class_num:
          count+=1
      
      
    elif class_num == 1:
      
      for i in array:
        if i == class_num:
          count+=1
          
    return count 
    
    
    
    
    
    
    
  
  # uses numpy to create a numpy array that will be used
  # *** NOTE: only yields accuracies etc. it is very fast! but does not map back to geotif
  def partition_np(X,y):
    X = X.flat_arr()
    y = y.flat_arr()
 
    
    
    #from sklearn.utils
    
    #var = input('You want some info? [y/n]?: ')
    #if var == 'y':
      
      #preproc.partition_log(X,y)
      
    
    
    # creating two individual sheets one for class 0 and another for class 1
    # Class 1...
    
    # preparing the sat image and label as one total array as pre because it still needs one last band... the spatial info
    pre_data = np.concatenate((X,y),axis=1)
    
    # adding an extra band with spatial location information
    N = preproc.xdim * preproc.ydim
    location = np.linspace(0,N,num=N,endpoint = True)
    location =np.reshape(location,[N,1])
    data = np.concatenate((pre_data,location),axis=1)
    
    #shuffle data to avoid bias and needed for parsing out label data from input sat data
    shuffle_data = sklearn.utils.resample(data, replace=False, random_state=None, stratify=None)
    
    # defining empty np arrays that will be used to add instances of class 1 or 0
    frame_X = np.zeros((X.shape[0],X.shape[1]))
    frame_y = np.zeros((y.shape[0],y.shape[1]))
    class_list = np.unique(y)
    
    # needed to append and make lists of only each class. class is -2 .... location is -1 in index
    class_0 = np.array([])
    class_1 = np.array([])
    stuff_0 = [i for i in data if i[-2] == 0]
    stuff_1 = [i for i in data if i[-2] == 1]
    # has to be np.stack not .append or .concatenate
    class_0 = np.stack(stuff_0, axis = 0)
    class_1 = np.stack(stuff_1, axis = 0)
    
    
    # will be used later... to be resampled to meet class 1 evenly
    shuffled_class_0 = sklearn.utils.resample(class_0, replace=False, 
                                           n_samples=None, random_state=None,
                                           stratify=None)
    
    
    #We need to use the exact same class_0 train and test data throught all partitions
    shuffled_class_1 = sklearn.utils.resample(class_1, replace=False, 
                                           n_samples=None, random_state=None,
                                           stratify=None)
    
    
    
    
    # writing partition tensor object 
    
    #using functions to get some useful info to automate the partition count
    num_partition = preproc.get_count(mineloc_img, class_num=0) // preproc.get_count(mineloc_img, class_num=1)
    # this is how many non_mine pixels need to be indexed at a time to get approx 
    # equal partitions of mine_pixels = 1 
    array_stride = preproc.get_count(mineloc_img, class_num=0) // num_partition  #158174 / 86 = 1839
    # now that some info is found we use it to create an emty tensor
    partition_np =np.empty([N,data.shape[1],num_partition])
    partition_np[:] = np.nan
 
    
    for partition in range(0,num_partition):
    #for partition in range(0,2):
      start = partition * array_stride
      #print(start)
      end = start + array_stride
      #print(end)
      array = shuffled_class_0[start:end,:]
      #print(array)
      #print()
      
      export_data = np.concatenate((shuffled_class_1,array))
      
      #final product of all the partitions
      partition_np[:export_data.shape[0],:,partition] = export_data[:,:]
    
    
    
    
    # creating file that will contain and store the partition tensor!
    
    os.chdir('/Users/josemarmolejo/Thesis/processing')
    #np.save('partitions',partition_np)
  
    
  
    
##############################################################################    
##############################################################################
##############################################################################
##############################################################################    
#################                   CLASS                      ###############
##############################################################################
##############################################################################    
#################                   MODEL                      ###############
##############################################################################
##############################################################################    
##############################################################################
##############################################################################
    
    
'''

The purpose of this class is to perform different statistical models and write 
results to .csv files 

It reads the packaged numpy binary.npy files and splits data into X_train, y_train
X_test, and y_test and performs statistic modeling!

'''
    
   
    
   
    
    
class model:
  
  
  # path to where the binary file is in computer
  def __init__(self, path):
    
   
    
    with open(path,'rb') as f:
      
      self.npfile = np.load(f)
      
      
    data = self.npfile
    class_1 = data[:preproc.get_count(mineloc_img, class_num=1),:,:]
    class_0 = data[preproc.get_count(mineloc_img, class_num=1):(1826 * 2 + 13),:,:] # *** FIXME there are 13 more *** 
    
    
    train_0, test_0 = sklearn.model_selection.train_test_split(class_0, test_size=.2, train_size=.8, 
                                                                              random_state=42, 
                                                                              shuffle=True, 
                                                                              stratify=None)
    
    
    train_1, test_1 = sklearn.model_selection.train_test_split(class_1,test_size=.2, train_size=.8,
                                                                              random_state=42,
                                                                              shuffle=True,
                                                                              stratify=None)
  
  
    # defining both training and testing X data as aa and bb to not use up X_train , X_test variable names
    aa = np.concatenate((train_1,train_0))   # this is actually X_train but with label and location bands
    bb = np.concatenate((test_1,test_0))    # this is actually X_test but with label, and location bands
    # now defining the y labels data
    self.y_train = aa[:,-2,:]
    self.y_test = bb[:,-2,:]
    # defining variable with location of each pixel
    self.train_loc = aa[:,-1,:]
    self.test_loc = bb[:,-1,:]
    #NOW finally defining final X_train and X_test data
    #
    self.X_train = aa[:,:16,:]
    self.X_test = bb[:,:16,:]
      
      
      
      
      
      
      
      
      
      
      
      
    
  #takes partition block binary.npy file as input and performs svc
  def svc():
    
    
    from sklearn.svm import SVC, LinearSVC

    N = file_np.npfile.shape[2]
    keyword = 'partition_'
    os.mkdir('SVC')
    os.chdir('/Users/josemarmolejo/Thesis/processing/SVC')
    #return 'FDSAFDSA'
    
    
    # Does the heavy SVC lifting 
    def svc_fn(N):
      
      #svc = SVC(C=100000.0, gamma=1e-07, random_state=0, kernel='rbf').fit(file_np.X_train[:,:,N], file_np.y_train[:,N])
      svc = SVC(C=1, gamma=1, random_state=0, kernel='rbf').fit(file_np.X_train[:,:,0], file_np.y_train[:,0])
      score = svc.score(file_np.X_test[:,:,N],file_np.y_test[:,N]) * 100
      predicted_label = svc.predict(file_np.X_test[:,:,N])
      decision_prob = svc.decision_function(file_np.X_test[:,:,N])
  
      # getting data / results per iteration each one can be accesed as function['score'], function['predicted_label'] etc.
      results = {
                        'score': score,
                        'predicted_label': predicted_label,
                        'decision_prob': decision_prob
                        }
      
      return results 
    
    
    #### creating a dictionary comprehension that will iteratively run SVC on each partion in binary partitions.npy
        # *** NOTE!!!: This_should_be_the_format = {keyword+str(n+1): svc_fn(n)['key'] for n in range(0,N)} ***
    
    
    
    # This function interprets svc prediction and parameters and exportrs to csv file for later use.
    def csv_data():
      
      for n in range(0,1): # FIXME: make N not 1!!!
        key = keyword + str(n + 1)
        arr = np.zeros([734,3])   # empty array framework
        
        # location
        arr[:,0] = file_np.test_loc[:,n].flatten()
        # prediction
        arr[:,1] = svc_fn(n)['predicted_label'].flatten()
        # label
        arr[:,2] = file_np.y_test[:,n].flatten()
        
        export = arr
        # print(key)
        # print(export)
        # print()
        # print()
        # print()
        
        export_df = pd.DataFrame(export, columns=['location','prediction','label'])
        #export_df.to_csv(key+'.csv')  # WRITE CSV !!!!
       
      
  
      # This is the score part of dictionary
    def score_svc():
      
      score_dict = {keyword+str(n+1): svc_fn(n)['score'] for n in range(0,N)}
      columns = ['score']
      export_score = pd.DataFrame.from_dict(score_dict,orient='index',columns=columns)  # just making that dictionary to pd
      export_score.to_csv("SVC_score.csv")   # exporting to processing directory
      
      return export_score
    
  
    
    ### executing the functions ###
    csv_data()
    score_svc()
          
    
    # ########################    DEPRECATED!!!!!!!!!!!!!! I left it though becasue it has good code! ################################
    # # this function just will just call the pixel location and make comparison of the predicted data and the correct labels   
    # def predicted_label():
    
    #   predicted_label_dict = {keyword+str(n+1): svc_fn(n)['predicted_label'] for n in range(0,N)}     # makes a dict of numpy array
    #   #columns = ['Predicted_label']
    #   #export_predicted_label = pd.DataFrame.from_dict(predicted_label_dict,orient='index')  # just making that dictionary to pd
    #   #return predicted_label_dict
    # ################################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
  def logreg():
    
    
    from sklearn.linear_model import LogisticRegression

    N = file_np.npfile.shape[2]   # num partitions
    keyword = 'partition_'
    os.mkdir('logREG')
    os.chdir('/Users/josemarmolejo/Thesis/processing/logREG')
    
    
    
    # Does the heavy logistic regression lifting 
    def logreg_fn(N):
      
      #svc = SVC(C=100000.0, gamma=1e-07, random_state=0, kernel='rbf').fit(file_np.X_train[:,:,N], file_np.y_train[:,N])
      logr = LogisticRegression(max_iter=1000,C=.1).fit(file_np.X_train[:,:,0], file_np.y_train[:,0])
      score = logr.score(file_np.X_test[:,:,N],file_np.y_test[:,N]) * 100
      predicted_label = logr.predict(file_np.X_test[:,:,N])
      decision_func = logr.decision_function(file_np.X_test[:,:,N])
      log_prob = logr.predict_log_proba(file_np.X_test[:,:,N])
  
      # getting data / results per iteration each one can be accesed as function['score'], function['predicted_label'] etc.
      results = {
                       'score': score,
                       'predicted_label': predicted_label,
                       'decision_func': decision_func,
                       'log_prob': log_prob
                       }
      
      return results 
    
    
    #### creating a dictionary comprehension that will iteratively run SVC on each partion in binary partitions.npy
        # *** NOTE!!!: This_should_be_the_format = {keyword+str(n+1): svc_fn(n)['key'] for n in range(0,N)} ***
    
    
    
    # This function interprets svc prediction and parameters and exportrs to csv file for later use.
    def csv_data():
      
      for n in range(0,N): # FIXME: make N not 1!!!
        key = keyword + str(n + 1)
        arr = np.zeros([734,3])   # empty array framework
        
        # location
        arr[:,0] = file_np.test_loc[:,n].flatten()
        # prediction
        arr[:,1] = logreg_fn(n)['predicted_label'].flatten()
        # label
        arr[:,2] = file_np.y_test[:,n].flatten()
        
        export = arr
        # print(key)
        # print(export)
        # print()
        # print()
        # print()
        
        export_df = pd.DataFrame(export, columns=['location','prediction','label'])
        #export_df.to_csv(key+'.csv')  # WRITE CSV !!!!
       
      
  
     # This is the score part of dictionary
    def score_logr():
      
      score_dict = {keyword+str(n+1): logreg_fn(n)['score'] for n in range(0,N)}
      columns = ['score']
      export_score = pd.DataFrame.from_dict(score_dict,orient='index',columns=columns)  # just making that dictionary to pd
      export_score.to_csv("logr_score.csv")   # exporting to processing directory
      
      return export_score
    
  
    
    ### executing the functions ###
    csv_data()
    score_logr()
          







    
    
  # class model function for principle component regression   
  def pcr():
    
    
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    N = file_np.npfile.shape[2]   # num partitions
    keyword = 'partition_'
    os.mkdir('PCR')
    os.chdir('/Users/josemarmolejo/Thesis/processing/PCR')
    
    
    
    # Does the heavy logistic regression lifting 
    def pcr_fn(N):
      
      pcr = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression()).fit(file_np.X_train[:,:,0], file_np.y_train[:,0])
      score = pcr.score(file_np.X_test[:,:,N],file_np.y_test[:,N]) * 100
      predicted_label = pcr.predict(file_np.X_test[:,:,N])
      #decision_func = logr.decision_function(file_np.X_test[:,:,N])
      #log_prob = logr.predict_log_proba(file_np.X_test[:,:,N])
  
      # getting data / results per iteration each one can be accesed as function['score'], function['predicted_label'] etc.
      results = {
                       'score': score,
                       'predicted_label': predicted_label,
                       #'decision_func': decision_func,
                       #'log_prob': log_prob
                       }
      
      return results 
    
    
    #### creating a dictionary comprehension that will iteratively run SVC on each partion in binary partitions.npy
        # *** NOTE!!!: This_should_be_the_format = {keyword+str(n+1): svc_fn(n)['key'] for n in range(0,N)} ***
    
    
    
    # This function interprets svc prediction and parameters and exportrs to csv file for later use.
    def csv_data():
      
      for n in range(0,N): # FIXME: make N not 1!!!
        key = keyword + str(n + 1)
        arr = np.zeros([734,3])   # empty array framework
        
        # location
        arr[:,0] = file_np.test_loc[:,n].flatten()
        # prediction
        arr[:,1] = pcr_fn(n)['predicted_label'].flatten()
        # label
        arr[:,2] = file_np.y_test[:,n].flatten()
        
        export = arr
        # print(key)
        # print(export)
        # print()
        # print()
        # print()
        
        export_df = pd.DataFrame(export, columns=['location','prediction','label'])
        export_df.to_csv(key+'.csv')  # WRITE CSV !!!!
       
      
  
     # This is the score part of dictionary
    def score_pcr():
      
      score_dict = {keyword+str(n+1): pcr_fn(n)['score'] for n in range(0,N)}
      columns = ['score']
      export_score = pd.DataFrame.from_dict(score_dict,orient='index',columns=columns)  # just making that dictionary to pd
      export_score.to_csv("pcr.csv")   # exporting to processing directory
      
      return export_score
    
  
    
    ### executing the functions ###
    csv_data()
    score_pcr()
    
    
    
    
    
    
  
    
    
    
    # class model function for principle component regression   
  def plsr():
    
    
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    

    N = file_np.npfile.shape[2]   # num partitions
    keyword = 'partition_'
    os.mkdir('PLSR') # FIXME! if this folder already exists it will still make a new one. make sure you do if else to check instead!
    os.chdir('/Users/josemarmolejo/Thesis/processing/PLSR')
    
    
    
    # Does the heavy logistic regression lifting 
    def plsr_fn(N):
      
      plsr = PLSRegression(n_components=2).fit(file_np.X_train[:,:,0], file_np.y_train[:,0])
      score = plsr.score(file_np.X_test[:,:,N],file_np.y_test[:,N]) * 100
      predicted_label = plsr.predict(file_np.X_test[:,:,N])
      #decision_func = logr.decision_function(file_np.X_test[:,:,N])
      #log_prob = logr.predict_log_proba(file_np.X_test[:,:,N])
  
      # getting data / results per iteration each one can be accesed as function['score'], function['predicted_label'] etc.
      results = {
                       'score': score,
                       'predicted_label': predicted_label,
                       #'decision_func': decision_func,
                       #'log_prob': log_prob
                       }
      
      return results 
    
    
    #### creating a dictionary comprehension that will iteratively run SVC on each partion in binary partitions.npy
        # *** NOTE!!!: This_should_be_the_format = {keyword+str(n+1): svc_fn(n)['key'] for n in range(0,N)} ***
    
    
    
    # This function interprets svc prediction and parameters and exportrs to csv file for later use.
    def csv_data():
      
      for n in range(0,N): # FIXME: make N not 1!!!
        key = keyword + str(n + 1)
        arr = np.zeros([734,3])   # empty array framework
        
        # location
        arr[:,0] = file_np.test_loc[:,n].flatten()
        # prediction
        arr[:,1] = plsr_fn(n)['predicted_label'].flatten()
        # label
        arr[:,2] = file_np.y_test[:,n].flatten()
        
        export = arr
        # print(key)
        # print(export)
        # print()
        # print()
        # print()
        
        export_df = pd.DataFrame(export, columns=['location','prediction','label'])
        export_df.to_csv(key+'.csv')  # WRITE CSV !!!!
       
      
  
     # This is the score part of dictionary
    def score_plsr():
      
      score_dict = {keyword+str(n+1): plsr_fn(n)['score'] for n in range(0,N)}
      columns = ['score']
      export_score = pd.DataFrame.from_dict(score_dict,orient='index',columns=columns)  # just making that dictionary to pd
      export_score.to_csv("plsr.csv")   # exporting to processing directory
      
      return export_score
    
  
    
    ### executing the functions ###
    csv_data()
    score_plsr()
    
    
    
    
    
    
    
  def linreg():
    

    from sklearn.linear_model import LinearRegression
    

    N = file_np.npfile.shape[2]   # num partitions
    keyword = 'partition_'
    os.mkdir('linreg') # FIXME! if this folder already exists it will still make a new one. make sure you do if else to check instead!
    os.chdir('/Users/josemarmolejo/Thesis/processing/linreg')
    
    
    
    # Does the heavy logistic regression lifting 
    def linreg_fn(N):
      
      linreg = LinearRegression().fit(file_np.X_train[:,:,0], file_np.y_train[:,0])
      score = linreg.score(file_np.X_test[:,:,N],file_np.y_test[:,N]) * 100
      predicted_label = linreg.predict(file_np.X_test[:,:,N])
      #decision_func = logr.decision_function(file_np.X_test[:,:,N])
      #log_prob = logr.predict_log_proba(file_np.X_test[:,:,N])
  
      # getting data / results per iteration each one can be accesed as function['score'], function['predicted_label'] etc.
      results = {
                       'score': score,
                       'predicted_label': predicted_label,
                       #'decision_func': decision_func,
                       #'log_prob': log_prob
                       }
      
      return results 
    
    
    #### creating a dictionary comprehension that will iteratively run SVC on each partion in binary partitions.npy
        # *** NOTE!!!: This_should_be_the_format = {keyword+str(n+1): svc_fn(n)['key'] for n in range(0,N)} ***
    
    
    
    # This function interprets svc prediction and parameters and exportrs to csv file for later use.
    def csv_data():
      
      for n in range(0,N): # FIXME: make N not 1!!!
        key = keyword + str(n + 1)
        arr = np.zeros([734,3])   # empty array framework
        
        # location
        arr[:,0] = file_np.test_loc[:,n].flatten()
        # prediction
        arr[:,1] = linreg_fn(n)['predicted_label'].flatten()
        # label
        arr[:,2] = file_np.y_test[:,n].flatten()
        
        export = arr
        # print(key)
        # print(export)
        # print()
        # print()
        # print()
        
        export_df = pd.DataFrame(export, columns=['location','prediction','label'])
        export_df.to_csv(key+'.csv')  # WRITE CSV !!!!
       
      
  
     # This is the score part of dictionary
    def score_linreg():
      
      score_dict = {keyword+str(n+1): linreg_fn(n)['score'] for n in range(0,N)}
      columns = ['score']
      export_score = pd.DataFrame.from_dict(score_dict,orient='index',columns=columns)  # just making that dictionary to pd
      export_score.to_csv("linreg.csv")   # exporting to processing directory
      
      return export_score
    
  
    
    ### executing the functions ###
    csv_data()
    score_linreg()
          















# I fucking hate tghis this is fucking stupid and I cant fsucking wait to b3 super far away from this bullshit.
# I just want to fucking rest and this little spoiled shit head is the dfuking worst.







  


#preproc
# insert file path
wv3_path = '/Users/josemarmolejo/Thesis/TM_fieldmaps/fixed_field_area.tif'
wv3_img = preproc(str(wv3_path))

mineloc_path = '/Users/josemarmolejo/Thesis/TM_fieldmaps/mine_locations_raster_1x1m.tif'
mineloc_img =  preproc(str(mineloc_path))


# model
file_path = '/Users/josemarmolejo/Thesis/processing/partitions.npy'
file_np = model(file_path) 
