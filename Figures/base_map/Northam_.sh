#!/bin/bash


gmt begin North_am_base_with_ocean

	gmt basemap -R235/280/24/41.25 -JM6i -B
	gmt set GMT_THEME cookbook
	gmt grdimage @earth_relief_01m -R235/280/24/41.25 -JM6i -I+d -t8
	gmt coast -R235/280/24/41.25 -JM6i -N1/thick,white -N2/0.75,black
	gmt coast -R235/280/24/41.25 -JM6i -EUS.NM+p2.0,brown -N1/.70,white,dotted # -Da -Gwheat -A5000 -Bg -Wfaint -EUS.NM+gbrown -Sazure3
	
	#gmt inset begin -DjTR+w1.5i+o0.15i/0.1i -F+gwhite+p1p+c0.1c+s
	gmt inset begin -DjTR+w1.5i+o0.15i/0.1i -F+gwhite+p1p+c0.1c+s
		gmt grdimage @earth_relief_10m -JG258/30N/? -I+d -t8 #-JM6i
		#gmt coast -Rg -JG258/30N/? -N1/.70,white -Da -Gwheat -A5000 -Bg -Wfaint -EUS.NM+p0.70,brown -Sazure3
		
		# for ocean DEM map
		gmt coast -Rg -JG258/30N/? -N1/.70,white -Da -Gwheat -A5000 -Bg -Wfaint -EUS.NMp0.70,brown
		gmt coast -Rg -JG258/30N/? -EUS.NM+p1.00,brown -N1/.70,white,dotted

	gmt inset end

gmt end show








