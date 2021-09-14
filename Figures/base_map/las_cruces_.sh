#!/bin/bash

gmt begin LC_area_base

	gmt basemap -R252.5/253.8/32/33 -JM6i -B
	gmt set GMT_THEME cookbook
	gmt grdimage @earth_relief_01s -R252.5/253.8/32/33 -JM6i -I+d -t2
	gmt coast -R252.5/253.8/32/33 -JM6i -N1/thicker,white -N2/1,black

gmt end show
