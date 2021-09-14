#!/bin/bash


# This shell script will generate a pdf of a mesh .nc file that was derived from a gtif that was converted using GDAL  
gmt begin pc_mesh

	gmt set GMT_THEME cookbook
	gmt grdimage test.nc #-Rg #-R235/280/24/41.25 
	
gmt end show