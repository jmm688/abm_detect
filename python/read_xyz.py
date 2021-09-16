#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:29:50 2021

@author: jose

Takes in a csv into pandas to then be export to numpy for np related operations.
take advantage of pandas better support for reading in csv files otherwise numpy reads really slow

oonce its in df, np is much quicker though :) 

"""
import pandas as pd
import numpy as np

csv_path = '/home/jose/Documents/LiDAR/TM/output_LAS/TM_clipped.csv'

points_df = pd.read_csv(csv_path)

points_np = points_df.to_numpy()