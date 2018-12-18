## Libraries
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import mode, norm

import cv2
import matplotlib.pyplot as plt
import progressbar
import pytesseract

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="Path to the input image")
parser.add_argument("-o", "--output", type=str, help="Path to the output image")
parser.add_argument("-d", "--direction", default='both+', type=str, choices=set(("light", "dark", "both", "both+")), help="Text searching")
parser.add_argument("-t", "--tesseract", action='store_true', help="Tesseract assistance")
parser.add_argument("-f", "--fulltesseract", action='store_true', help="Full Tesseract")
args = vars(parser.parse_args())
IMAGE_PATH = args["image"]
OUTPUT_PATH = args["output"]
DIRECTION = args["direction"]
TESS = args["tesseract"]
FULL_OCR = args["fulltesseract"]

## Parameters
AREA_LIM = 2.0e-4
PERIMETER_LIM = 1e-4
ASPECT_RATIO_LIM = 5.0
OCCUPATION_LIM = (0.23, 0.90)
COMPACTNESS_LIM = (3e-3, 1e-1)
SWT_TOTAL_COUNT = 10
SWT_STD_LIM = 20.0
STROKE_WIDTH_SIZE_RATIO_LIM = 0.02			## Min value
STROKE_WIDTH_VARIANCE_RATIO_LIM = 0.15		## Min value
STEP_LIMIT = 10
KSIZE = 3
ITERATION = 7
MARGIN = 10

## Displaying function
def pltShow(*images):
	count = len(images)
	nRow = np.ceil(count / 3.)
	for i in range(count):
		plt.subplot(nRow, 3, i + 1)
		if len(images[i][0].shape) == 2:
			plt.imshow(images[i][0], cmap='gray')
		else:
			plt.imshow(images[i][0])
		plt.xticks([])
		plt.yticks([])
		plt.title(images[i][1])
	plt.show()

class TextDetection(object):

	def __init__(self, image_path):
		self.imagaPath = image_path
		img = cv2.imread(image_path)
		rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.img = rgbImg
		self.final = rgbImg.copy()
		self.height, self.width = self.img.shape[:2]
		self.grayImg = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
		self.cannyImg = self.applyCanny(self.img)
		self.sobelX = cv2.Sobel(self.grayImg, cv2.CV_64F, 1, 0, ksize=-1)
		self.sobelY = cv2.Sobel(self.grayImg, cv2.CV_64F, 0, 1, ksize=-1)
		self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
		self.stepsY = self.sobelX.astype(int)
		self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
		self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
		self.gradsY = self.stepsY / (self.magnitudes + 1e-10)

	def getMSERegions(self, img):
		mser = cv2.MSER_create()
		# img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
		regions, bboxes = mser.detectRegions(img)
		return regions, bboxes

	def colorRegion(self, img, region):
		img[region[:, 1], region[:, 0], 0] = np.random.randint(low=100, high=256)
		img[region[:, 1], region[:, 0], 1] = np.random.randint(low=100, high=256)
		img[region[:, 1], region[:, 0], 2] = np.random.randint(low=100, high=256)
		return img

	def applyCanny(self, img, sigma=0.33):
		v = np.median(img)
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		return cv2.Canny(img, lower, upper)

	def getRegionShape(self, region):
		return (max(region[:, 1]) - min(region[:, 1]), max(region[:, 0]) - min(region[:, 0]))

	def getRegionArea(self, region):
		return len(list(region))		## Number of pixels

	def getRegionPerimeter(self, region):
		x, y, w, h = cv2.boundingRect(region)
		return len(np.where(self.cannyImg[y:y + h, x:x + w] != 0)[0])

	def getOccupyRate(self, region):
		return (1.0 * self.getRegionArea(region)) / (self.getRegionShape(region)[0] * self.getRegionShape(region)[1] + 1.0e-10)

	def getAspectRatio(self, region):
		return (1.0 * max(self.getRegionShape(region))) / (min(self.getRegionShape(region)) + 1e-4)

	def getCompactness(self, region):
		return (1.0 * self.getRegionArea(region)) / (1.0 * self.getRegionPerimeter(region) ** 2)

	def getSolidity(self, region):
		x, y, w, h = cv2.boundingRect(region)
		return (1.0 * self.getRegionArea(region)) / ((1.0 * w * h) + 1e-10)

	def getStrokeProperties(self, strokeWidths):
		if len(strokeWidths) == 0:
			return (0, 0, 0, 0, 0, 0)
		try:
			mostStrokeWidth = mode(strokeWidths, axis=None)[0][0]	## Most probable stroke width is the most one
			mostStrokeWidthCount = mode(strokeWidths, axis=None)[1][0]	## Most probable stroke width is the most one
		except IndexError:
			mostStrokeWidth = 0
			mostStrokeWidthCount = 0
		try:
			mean, std = norm.fit(strokeWidths)
			xMin, xMax = int(min(strokeWidths)), int(max(strokeWidths))
		except ValueError:
			mean, std, xMin, xMax = 0, 0, 0, 0
		return (mostStrokeWidth, mostStrokeWidthCount, mean, std, xMin, xMax)

	def getStrokes(self, xywh):
		# strokes = np.zeros(self.grayImg.shape)
		x, y, w, h = xywh
		strokeWidths = np.array([[np.Infinity, np.Infinity]])
		for i in range(y, y + h):
			for j in range(x, x + w):
				if self.cannyImg[i, j] != 0:
					stepX = self.stepsX[i, j]
					stepY = self.stepsY[i, j]
					gradX = self.gradsX[i, j]
					gradY = self.gradsY[i, j]

					prevX, prevY, prevX_opp, prevY_opp, stepSize = i, j, i, j, 0

					if DIRECTION == "light":
						go, go_opp = True, False
					elif DIRECTION == "dark":
						go, go_opp = False, True
					else:
						go, go_opp = True, True

					strokeWidth = np.Infinity
					strokeWidth_opp = np.Infinity
					while (go or go_opp) and (stepSize < STEP_LIMIT):
						stepSize += 1

						if go:
							curX = np.int(np.floor(i + gradX * stepSize))
							curY = np.int(np.floor(j + gradY * stepSize))
							if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
								go = False
							if go and ((curX != prevX) or (curY != prevY)):
								try:
									if self.cannyImg[curX, curY] != 0:
										if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[curX, curY]) < np.pi/2.0:
											strokeWidth = int(np.sqrt((curX - i) ** 2  + (curY - j) ** 2))

											go = False
								except IndexError:
									go = False

								prevX = curX
								prevY = curY

						if go_opp:
							curX_opp = np.int(np.floor(i - gradX * stepSize))
							curY_opp = np.int(np.floor(j - gradY * stepSize))
							if (curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w):
								go_opp = False
							if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
								try:
									if self.cannyImg[curX_opp, curY_opp] != 0:
										if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[curX_opp, curY_opp]) < np.pi/2.0:
											strokeWidth_opp = int(np.sqrt((curX_opp - i) ** 2  + (curY_opp - j) ** 2))

											go_opp = False

								except IndexError:
									go_opp = False

								prevX_opp = curX_opp
								prevY_opp = curY_opp

					strokeWidths = np.append(strokeWidths, [(strokeWidth, strokeWidth_opp)], axis=0)

		strokeWidths_opp =  np.delete(strokeWidths[:, 1], np.where(strokeWidths[:, 1] == np.Infinity))
		strokeWidths = 		np.delete(strokeWidths[:, 0], np.where(strokeWidths[:, 0] == np.Infinity))
		return strokeWidths, strokeWidths_opp

	def detect(self):
		res10 = np.zeros_like(self.img)
		boxRes = self.img.copy()

		regions, bboxes = self.getMSERegions(self.grayImg)

		n1 = len(regions)
		n2, n3, n4, n5, n6, n7, n8, n9, n10 = [0] * 9
		bar = progressbar.ProgressBar(maxval=n1, widgets=[progressbar.Bar(marker='=', left='[', right=']'), ' ', progressbar.SimpleProgress()])

		bar.start()
		## Coloring the regions
		for i, region in enumerate(regions):
			bar.update(i + 1)

			if self.getRegionArea(region) > self.grayImg.shape[0] * self.grayImg.shape[1] * AREA_LIM:
				n2 += 1

				if self.getRegionPerimeter(region) > 2 * (self.grayImg.shape[0] + self.grayImg.shape[1]) * PERIMETER_LIM:
					n3 += 1

					if self.getAspectRatio(region) < ASPECT_RATIO_LIM:
						n4 += 1

						if (self.getOccupyRate(region) > OCCUPATION_LIM[0]) and (self.getOccupyRate(region) < OCCUPATION_LIM[1]):
							n5 += 1

							if (self.getCompactness(region) > COMPACTNESS_LIM[0]) and (self.getCompactness(region) < COMPACTNESS_LIM[1]):
								n6 += 1

								# x, y, w, h = cv2.boundingRect(region)
								x, y, w, h = bboxes[i]

								# strokeWidths, strokeWidths_opp, strokes = self.getStrokes((x, y, w, h))
								strokeWidths, strokeWidths_opp = self.getStrokes((x, y, w, h))
								if DIRECTION != "both+":
									strokeWidths = np.append(strokeWidths, strokeWidths_opp, axis=0)
									strokeWidth, strokeWidthCount, mean, std, xMin, xMax = self.getStrokeProperties(strokeWidths)
								else:
									strokeWidth, strokeWidthCount, mean, std, xMin, xMax = self.getStrokeProperties(strokeWidths)
									strokeWidth_opp, strokeWidthCount_opp, mean_opp, std_opp, xMin_opp, xMax_opp = self.getStrokeProperties(strokeWidths_opp)
									if strokeWidthCount_opp > strokeWidthCount:		## Take the strokeWidths with max of counts strokeWidth (most probable one)
										strokeWidths = strokeWidths_opp
										strokeWidth = strokeWidth_opp
										strokeWidthCount = strokeWidthCount_opp
										mean = mean_opp
										std = std_opp
										xMin = xMin_opp
										xMax = xMax_opp

								if len(strokeWidths) > SWT_TOTAL_COUNT:
									n7 += 1

									if std < SWT_STD_LIM:
										n8 += 1

										strokeWidthSizeRatio = strokeWidth / (1.0 * max(self.getRegionShape(region)))
										if strokeWidthSizeRatio > STROKE_WIDTH_SIZE_RATIO_LIM:
											n9 += 1

											strokeWidthVarianceRatio = (1.0 * strokeWidth) / (std ** std)
											if strokeWidthVarianceRatio > STROKE_WIDTH_VARIANCE_RATIO_LIM:
												n10 += 1
												res10 = self.colorRegion(res10, region)

		bar.finish()
		print("{} regions left.".format(n10))

		## Binarize regions
		binarized = np.zeros_like(self.grayImg)
		rows, cols, color = np.where(res10 != [0, 0, 0])
		binarized[rows, cols] = 255

		## Dilate regions and find contours
		kernel = np.zeros((KSIZE, KSIZE), dtype=np.uint8)
		kernel[(KSIZE // 2)] = 1

		if TESS:
			print("Tesseract eliminates..")

		res = np.zeros_like(self.grayImg)
		dilated = cv2.dilate(binarized.copy(), kernel, iterations=ITERATION)
		image, contours, hierarchies = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for i, (contour, hierarchy) in enumerate(zip(contours, hierarchies[0])):
			if hierarchy[-1] == -1:

				if TESS:
					x, y, w, h = cv2.boundingRect(contour)
					if (y - MARGIN > 0) and (y + h + MARGIN < self.height) and (x - MARGIN > 0) and (x + w + MARGIN < self.width):
						cv2.imwrite("text.jpg", self.final[y - MARGIN:y + h + MARGIN, x - MARGIN:x + w + MARGIN])
					else:
						cv2.imwrite("text.jpg", self.final[y:y + h, x:x + w])

					###################
					## Run tesseract ##
					###################
					string = pytesseract.image_to_string(Image.open("text.jpg"))
					if string is not u'':
						rect = cv2.minAreaRect(contour)
						box = cv2.boxPoints(rect)
						box = np.int0(box)
						cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
						cv2.drawContours(res, [box], 0, 255, -1)
					os.remove("text.jpg")

				else:
					rect = cv2.minAreaRect(contour)
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
					cv2.drawContours(res, [box], 0, 255, -1)

		return res

	def fullOCR(self):
		bounded = self.img.copy()
		H, W = self.height, self.width
		res = np.zeros_like(self.grayImg)

		string = pytesseract.image_to_string(Image.open(self.imagaPath))
		if string == u'':
			return bounded, res

		boxes = pytesseract.image_to_boxes(Image.open(self.imagaPath))
		boxes = [map(int, i) for i in [b.split(" ")[1:-1] for b in boxes.split("\n")]]

		for box in boxes:
			b = (int(box[0]), int(H - box[1]), int(box[2]), int(H - box[3]))
			cv2.rectangle(bounded, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
			cv2.rectangle(res, (b[0], b[1]), (b[2], b[3]), 255, -1)

		# pltShow((img, "Original"), (bounded, "Boxes"), (res, "Mask"))
		return bounded, res

if IMAGE_PATH:
	td = TextDetection(IMAGE_PATH)
	if FULL_OCR:
		bounded, res = td.fullOCR()
		pltShow((td.img, "Original"), (bounded, "Final"), (res, "Mask"))
	else:
		res = td.detect()
		pltShow((td.img, "Original"), (td.final, "Final"), (res, "Mask"))
		if OUTPUT_PATH:
			plt.imsave(OUTPUT_PATH, td.final)
			print("{} saved".format(OUTPUT_PATH))
