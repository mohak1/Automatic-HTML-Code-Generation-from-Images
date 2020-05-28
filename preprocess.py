import cv2
import numpy as np
import os
from PIL import Image

#this method runs for a single image
def preprocessing(path_to_image):
	img = cv2.imread(path_to_image)
	#resize all the images to a fixed size (width=950, height=1000)
	#resize if image is larger than fixed size
	#give a background if its smaller than fixed size
	height = np.size(img, 0)
	width = np.size(img, 1)
	if(height<1000 and width<950):
		#give a black background
		#create black image
		height = 1000
		width = 950
		black_image = np.zeros((height,width,3), np.uint8)
		black_image[:, 0:width] = (0,0,0)
		x_offset = int((width - img.shape[1])/2)
		y_offset = int((height - img.shape[0])/2)		
		black_image[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
		img=black_image
		# cv2.imshow('with background',img)
		# cv2.waitKey(0)
	else:
		#resize the image
		img = cv2.resize(img, (950, 1000))	
	
	#grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#dilation
	#kernel for dialation
	kernel = np.ones((2,2), np.uint8)	
	img_dilation = cv2.dilate(img_gray, kernel, iterations=1)
	#thresholding 
	ret, thresh = cv2.threshold(img_dilation, 120, 255, cv2.THRESH_BINARY) 
	#canny edge detection
	edges = cv2.Canny(img_dilation,100,101)

	#create a temporary processed file
	#give a path (preferably in the same directory) for storing the temp file.
	#this is the file generated after preprocessing and is deleted automatically 
	#after detectoin of objects from it.
	path='PATH_TO_SAVE_TEMP_IMAGE/temp.jpg'
	cv2.imwrite(path,edges)
	return path
