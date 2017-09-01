#!/usr/bin/env python

"""
Author: Manpreet Singh 2016
Headless mode for raspberry pi.
This program only returns the X and Y position of an object if it is found in the image
"""

import cv2 as cv
import numpy as np
from networktables import NetworkTable

from Template import Template

nt = NetworkTable.getTable("VISION")
template = Template()
cap = cv.VideoCapture(1)

while True:
	_, im = cap.read()
	im = np.array(im, dtype=np.uint8)	
	hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
	
	# HSV Ranges, hard coded values
	lower_limit = np.array([78, 176, 0])
	upper_limit = np.array([119, 255, 255])
		
	mask = cv.inRange(hsv, lower_limit, upper_limit)
	
	_, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
		
	if len(contours) > 0:

		template.best_match(contours)
		
		# Access object for contour data
		moments = cv.moments(cnt)
		if moments['m00'] != 0:
			# Get center of contour coordinates
			centroid_x = int(moments['m10'] / moments['m00'])
			centroid_y = int(moments['m01'] / moments['m00'])
			
			nt.putNumber('centerX', centroid_x)
			nt.putNumber('centerY', centroid_y)
			
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
