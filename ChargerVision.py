#!/usr/bin/env python

"""
Author: Manpreet Singh 2016
GUI program for Calibration
This Program is meant to run on a computer to find the lower/upper HSV Values for calibration
"""

from pprint import pprint

import cv2 as cv
import numpy as np
# from networktables import NetworkTables

from Template import Template


def nothing(foo):
	pass

# NetworkTables.initialize()
# nt = NetworkTables.getTable("VISION")
# target_template = Template()
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_EXPOSURE, -13)

cv.namedWindow('Control')
cv.createTrackbar('Lower_Hue', 'Control', 0, 255, nothing)
cv.createTrackbar('Upper_Hue', 'Control', 0, 255, nothing)

cv.createTrackbar('Lower_Sat', 'Control', 0, 255, nothing)
cv.createTrackbar('Upper_Sat', 'Control', 0, 255, nothing)

cv.createTrackbar('Lower_Vib', 'Control', 0, 255, nothing)
cv.createTrackbar('Upper_Vib', 'Control', 0, 255, nothing)

cv.createTrackbar('Contour', 'Control', 0, 12, nothing)

switch = 'Track Object \n0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'Control', 0, 1, nothing)

font = cv.FONT_HERSHEY_SIMPLEX

kernel = np.ones((5,5), np.uint8)

while True:
	_, im = cap.read()
	im = np.array(im, dtype=np.uint8)
	hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
	
	lower_hue = cv.getTrackbarPos('Lower_Hue', 'Control')
	upper_hue = cv.getTrackbarPos('Upper_Hue', 'Control')
	
	lower_sat = cv.getTrackbarPos('Lower_Sat', 'Control')
	upper_sat = cv.getTrackbarPos('Upper_Sat', 'Control')
	
	lower_vib = cv.getTrackbarPos('Lower_Vib', 'Control')
	upper_vib = cv.getTrackbarPos('Upper_Vib', 'Control')
	
	contour_num = cv.getTrackbarPos('Contour', 'Control')
	
	switch_val = cv.getTrackbarPos(switch, 'Control')
	
	# for testing
	lower_limit = np.array([lower_hue, lower_sat, lower_vib])
	upper_limit = np.array([upper_hue, upper_sat, upper_vib])
	
	im_thresh = cv.inRange(hsv, lower_limit, upper_limit)
	im_erode = cv.erode(im_thresh, kernel, iterations = 2)
	im_mask = cv.dilate(im_erode, kernel, iterations = 1)

	# im_mask = cv.inRange(hsv, lower_limit, upper_limit)

	
	im_res = cv.bitwise_and(im, im, mask=im_mask)
	
	contours, hierarchy = cv.findContours(im_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	cv.drawContours(im_res, contours, -1, (255, 0, 255), 1)
		
	if switch_val == 1 and len(contours) > 0: 

		# cnt = contours[target_template.best_match(contours)]
		sorted_cnt = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
		cnt = sorted_cnt[contour_num]

		# Access object for contour data
		moments = cv.moments(cnt)
		if moments['m00'] != 0:
			# Get center of contour coordinates
			centroid_x = int(moments['m10'] / moments['m00'])
			centroid_y = int(moments['m01'] / moments['m00'])
			
			# nt.putNumber('centerX', centroid_x)
			# nt.putNumber('centerY', centroid_y)
			
			# Print the Coordinates onto image
			x_t = 'x:%s' % centroid_x
			y_t = 'y:%s' % centroid_y
			angle = 'a:%s' % (480 / (68.5 * np.sin(np.arctan(.75))) * centroid_y + 480)
			
			cv.putText(im_res, x_t, (10, 40), font, 1, (255, 0, 255), 0)
			cv.putText(im_res, y_t, (10, 70), font, 1, (255, 255, 0), 0)
			cv.putText(im_res, angle, (10, 100), font, 1, (255, 255, 255), 0)
			
			# Draw dot at the center of the Contour
			cv.circle(im_res, (centroid_x, centroid_y), 2, (255, 0, 255), 1)
			# Draw Bounding Rectangle around Contour
			x, y, w, h = cv.boundingRect(cnt)
			cv.rectangle(im_res, (x, y), (x + w, y + h), (0, 255, 0), 1)

		else:
			print('No Info')
			print(len(sorted_cnt))


	# Display output windows
	cv.imshow('Mask', im_mask)
	cv.imshow('Output', im_res)
	cv.imshow('hsv', hsv)

	if cv.waitKey(1) & 0xFF == ord('q'):
		cv.destroyAllWindows()
		# print out the list of matched contours, ordered from the most likely to least likely
		# pprint(target_template.list_of_matched(contours))
		break

cap.release()
