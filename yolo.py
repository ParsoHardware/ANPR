# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

####
import imutils
import pytesseract

#import pyocr
#import pyocr.builders

from PIL import Image

####
#Time for all algorithm
startAlg = time.time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])

#Resize image
# size = 600
# r = image.shape[1] / image.shape[0]
# dim = (int(size * r), size)
# image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
print("[INFO] Starting YOLO algorithm...")
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

plates = []
# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		#Cropping the plate
		if classIDs[i] == 0:
			cropped = image[y:(y+h) , x:(x+w)]
			plates.append(cropped)
			#cv2.imshow("Plate" , cropped)
		# draw a bounding box rectangle and label on the image
		#color = [int(c) for c in COLORS[classIDs[i]]]
		#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if plates != []:
	count = 1
	for plate in plates:
		gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) #convert to grey scale
		blurred = cv2.bilateralFilter(gray, 11, 41, 41) #Blur to reduce noise
		#edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
		(T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
		
		coords = np.column_stack(np.where(thresh > 0))
		angle = cv2.minAreaRect(coords)[-1]

		# the `cv2.minAreaRect` function returns values in the
		# range [-90, 0); as the rectangle rotates clockwise the
		# returned angle trends to 0 -- in this special case we
		# need to add 90 degrees to the angle
		if angle < -45:
			angle = -(90 + angle)

		# otherwise, just take the inverse of the angle to make
		# it positive
		else:
			angle = -angle

		print("[INFO] Rotation angle:"+str(angle))

		(h, w) = thresh.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0) #Rotate image
		#thresh = cv2.warpAffine(thresh, M, (w, h))
		thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		#blurred = cv2.bilateralFilter(thresh, 11, 17, 17) #Blur to reduce noise
		
		config = ("-l spa --oem 1 --psm 4")
		text = pytesseract.image_to_string(thresh, config=config)
		print("[INFO] Detected Number is:",text)
		
		win = "Plate_N" + str(count) + ".jpg"
		#out = np.hstack([thresh])
		cv2.imshow(win,thresh)
		cv2.imwrite(win, thresh)
		count = count + 1

# if plates != []:
# 	count = 1
# 	for plate in plates:
# 		text = "Plate_N" + str(count) + ".jpg"
# 		cv2.imwrite(text, plate)
# 		#cv2.imshow(text , plate)
# 		count = count + 1

# show algorithm timing information on YOLO
endAlg = time.time()
print("[INFO] Complete algorithm took {:.6f} seconds".format(endAlg - startAlg))

cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.waitKey(0)
# resize image 
# size = 600
# r = image.shape[1] / image.shape[0]
# dim = (int(size * r), size)
# image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# outputFile = args.image[:-4]+'_yolo_out.jpg'
# cv.imwrite(outputFile, frame.astype(np.uint8))
