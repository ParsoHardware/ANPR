# USAGE
# python anpr.py --image images/baggage_claim.jpg

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
##
import imutils
import pytesseract

#import pyocr
#import pyocr.builders

from PIL import Image


startAll = time.time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
#ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
#ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load our input image and grab its spatial dimensions
img = cv2.imread(args["image"])
#img = cv2.imread('4.jpg',cv2.IMREAD_COLOR)

#img = cv2.resize(img, (620,480) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
 # approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.018 * peri, True)

 # if our approximated contour has four points, then
 # we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break



if screenCnt is None:
	detected = 0
	print( "No contour detected" )
else:
	detected = 1

if detected == 1:
	cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx+11:bottomx-10, topy+1:bottomy-1]

#blurred

(T, thresh) = cv2.threshold(Cropped, 155, 255, cv2.THRESH_BINARY)

blurred = cv2.bilateralFilter(thresh, 11, 17, 17) #Blur to reduce noise

#Read the number plate
#text = pytesseract.image_to_string(thresh, config='-l eng --oem 3 --psm 12')
text = pytesseract.image_to_string(blurred, config='')
#text = pytesseract.image_to_string(blurred, config='-l eng --oem 3 --psm 12')

#tools = pyocr.get_available_tools()[0]
#text = tools.image_to_string(Image.open(Cropped), builder=pyocr.builders.DigitBuilder())

print("Detected Number is:",text)

######################
endAll = time.time()
print("[INFO] All Algorithm took {:.6f} seconds".format(endAll - startAll))

#cv2.imshow('image',img)
out = np.hstack([Cropped, blurred, thresh])
cv2.imshow('OUTPUT',out)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


