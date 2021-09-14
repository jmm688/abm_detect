#!/bin/bash


gmt begin NM_base

	gmt basemap -R250/258/31/37.5 -JM6i -B
	gmt set GMT_THEME cookbook
	gmt grdimage @earth_relief_15s -R250/258/31/37.5 -JM6i -I+d -t2
	gmt coast -R250/258/31/37.5 -JM6i -N1/thicker,white -N2/1,black
	gmt coast -R250/258/31/37.5 -JM6i -EUS.NM+p2.5,brown -N1/thicker,white,dotted

	gmt inset begin -DjTR+w2i/0.93i+o0.15i/0.1i -F+gwhite+p1p+c0.1c+s
		#gmt grdimage @earth_relief_10m -JG258/30N/? -I+d -t8 #-JM6i
		gmt coast -R235/280/24/41.25 -JM? -N1/.70,white -N2/0.001, -Da -Gwheat -A5000 -Wfaint -Sazure3
		gmt coast -R235/280/24/41.25 -JM? -EUS.NM+p1.00,brown -N1/.70,white,dotted
		
		# for ocean DEM map
		#gmt coast -R235/280/24/41.25 -JM? -N1/.70,white -Da -Gwheat -A5000 -Bg -Wfaint -EUS.NMp0.70,brown
	gmt inset end


gmt end show




