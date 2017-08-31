#!/usr/bin/env python

import cv2
import numpy as np


# Template Matching Class
class Template:

	# Set up the template contour for later use
	def __init__(self):
		self.__template_image__ = cv2.imread("template.png")
		self.__template_image__ = cv2.cvtColor(self.__template_image__, cv2.COLOR_BGR2HSV)
		self.__template_image__ = cv2.inRange(self.__template_image__, np.array([0, 0, 2]), np.array([255, 255, 255]))
		_, contours, h = cv2.findContours(self.__template_image__, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		mx_indx = np.argmax(areas)
		self.__template_contour__ = contours[mx_indx]

	# Compare the list of contours to the template and sort by the closest matching contours.
	# Returns a 2d array, first column contains a number representing the match, smaller the number,
	# the closer the contour is to the template, second column represents the array index in the
	# originally passed list
	def list_of_matched(self, contours):
		list_of_matches = []
		index = 0
		for c in contours:
			list_of_matches.append([cv2.matchShapes(self.__template_contour__, c, 1, 0.0), index])
			index += 1
		list_of_matches = sorted(list_of_matches, key=lambda x: x[0])
		return list_of_matches

	# Get the index of the best matched contour in passed list of contours
	def best_match(self, contours):
		return self.list_of_matched(contours)[0][1]

	# For debugging, Just to see what the template looks like
	def get_image_to_show(self):
		return self.__template_image__

