import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, mode
from PIL import Image
import pytesseract
import argparse, progressbar, sys, math

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
parser.add_argument("-d", "--direction", required=True, type=str, choices=set(("light", "dark", "both", "both+")), help="Text searching")
parser.add_argument("--deskew", action='store_true', help="Deskewing")
parser.add_argument("-t", "--tesseract", action='store_true', help="Deskewing")
args = vars(parser.parse_args())
IMAGE_PATH = args["image"]
DIRECTION = args["direction"]
DESKEW = args["deskew"]
TESS = args["tesseract"]

AREA_LIM = 1.0e-4
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
MARGIN = 3
SAVE = False

img = cv2.imread(IMAGE_PATH)
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
	if SAVE:
		plt.savefig("test/final.jpg")
	plt.show()

def applyKmeans(img, kVal, ATTEMPT=10):
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1.0)
	data = img.reshape((-1, 3))
	data = np.float32(data)
	if cv2.__version__.startswith("2"):
		compactness, labels, centers = cv2.kmeans(data, kVal, criteria, ATTEMPT, cv2.KMEANS_PP_CENTERS)
	else:
		compactness, labels, centers = cv2.kmeans(data, kVal, None, criteria, ATTEMPT, cv2.KMEANS_PP_CENTERS)
	centers = np.uint8(centers)
	kRes = centers[labels.flatten()]
	kRes = kRes.reshape((img.shape))
	return kRes

def applyCloseEdgeDetect(img):
	res = img.copy()
	quantized = img
	# quantized = applyKmeans(rgbImg, 5)
	grayImg = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)

	canny = cv2.Canny(grayImg, 120, 200)
	# sobelX = cv2.Sobel(grayImg, cv2.CV_8U, 1, 0, ksize=3)
	# sobelY = cv2.Sobel(grayImg, cv2.CV_8U, 0, 1, ksize=3)
	# sobelX = cv2.convertScaleAbs(sobelX)
	# sobelY = cv2.convertScaleAbs(sobelY)
	# sobel = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0);

	ret, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
	morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	if cv2.__version__.startswith("2"):
		contours, hierarchy = cv2.findContours(morphed, 0, 1)
	else:
		im, contours, hierarchy = cv2.findContours(morphed, 0, 1)
	for contour in contours:
		if len(contour) > 100:
			contours_poly = cv2.approxPolyDP(contour, 3, True)
			x, y, w, h = cv2.boundingRect(contours_poly)
			cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

	pltShow(img, quantized, canny, morphed, res)

def applyGradientDetect(img):
	res = img.copy()
	quantized = img
	# quantized = applyKmeans(rgbImg, 5)
	grayImg = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(grayImg, cv2.MORPH_GRADIENT, kernel)

	ret, thresh = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
	connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	if cv2.__version__.startswith("2"):
		contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	else:
		im, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
		contour_poly = cv2.approxPolyDP(contour, 3, True)
		x, y, w, h = cv2.boundingRect(contour_poly)
		cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

	pltShow(img, quantized, grad, connected, res)

class TextDetection(object):

	def __init__(self, img):
		# self.img = cv2.resize(img, (int(0.1 * img.shape[0]), int(0.1 * img.shape[1])), interpolation=cv2.INTER_CUBIC)
		self.img = img
		self.height, self.width = self.img.shape[:2]
		self.final = img.copy()
		self.kVal = 5
		self.quantized = applyKmeans(self.img.copy(), self.kVal)  ## Applying color quantization before with k-means, helps MSER!! Find a good value for K!!
		self.grayImg = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
		# self.cannyImg = [cv2.Canny(self.img, 50, 200), cv2.Canny(self.quantized, 50, 200)]
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

	def saveRegions(self, img):
		regions, bboxes = self.getMSERegions(img)
		for i, region in enumerate(regions):
			res = np.zeros_like(img)
			boxRes = img.copy()
			self.colorRegion(res, region)
			# x, y, w, h = cv2.boundingRect(region)
			# cv2.rectangle(boxRes, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(boxRes, (bboxes[i][0], bboxes[i][1]), (bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3]), (0, 255, 0), 2)
			cv2.imwrite("regions/{}_region.jpg".format(i), res)
			cv2.imwrite("regions/{}_region_box.jpg".format(i), boxRes)

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
		# epsilon = 0.1 * self.getRegionPerimeter(region)
		# convexRegion = cv2.approxPolyDP(region, 1, True)
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
		x, y, w, h = xywh
		# strokes = np.zeros(self.grayImg.shape)
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

											# cv2.line(strokes, (j, i), (curY, curX), (255, 255, 255), 1)
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

											# cv2.line(strokes, (j, i), (curY_opp, curX_opp), (120, 120, 120), 1)
											go_opp = False

								except IndexError:
									go_opp = False

								prevX_opp = curX_opp
								prevY_opp = curY_opp

					strokeWidths = np.append(strokeWidths, [(strokeWidth, strokeWidth_opp)], axis=0)

		strokeWidths_opp =  np.delete(strokeWidths[:, 1], np.where(strokeWidths[:, 1] == np.Infinity))
		strokeWidths = 		np.delete(strokeWidths[:, 0], np.where(strokeWidths[:, 0] == np.Infinity))
		return strokeWidths, strokeWidths_opp
		# return strokeWidths, strokeWidths_opp, strokes

	def deskew(self, img):

		if img.shape[-1] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		coords = np.column_stack(np.where(img > 0))
		angle = cv2.minAreaRect(coords)[-1]

		if angle < -45:
			angle = -(90 + angle)
		else:
			angle = -angle

		# Rotate
		(h, w) = img.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		return rotated, angle

	def rotateRect(self, x, y, w, h, angle):
		newPoints = []
		radius = math.sqrt((x - w / 2)**2 + (y - h / 2)**2)
		center = (x + w / 2, y + h / 2)
		points = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
		for point in points:
			angle += math.atan2(point[1] - center[1], point[0] - center[0])
			newPoints.append( (round(center[0] + radius * math.cos(angle)), round(center[1] + radius * math.sin(angle))) )

		return newPoints

	def detect(self):
		res  = np.zeros_like(self.img)
		res2 = np.zeros_like(self.img)
		res3 = np.zeros_like(self.img)
		res4 = np.zeros_like(self.img)
		res5 = np.zeros_like(self.img)
		res6 = np.zeros_like(self.img)
		res7 = np.zeros_like(self.img)
		res8 = np.zeros_like(self.img)
		res9 = np.zeros_like(self.img)
		res10 = np.zeros_like(self.img)
		boxRes = self.img.copy()

		# regions1, bboxes1 = self.mser.detectRegions(self.quantized)
		regions2, bboxes2 = self.getMSERegions(self.grayImg)
		regions, bboxes = regions2, bboxes2
		# regions = np.concatenate((regions1, regions2))
		# bboxes = np.concatenate((bboxes1, bboxes2))

		n1 = len(regions)
		n2, n3, n4, n5, n6, n7, n8, n9, n10 = [0] * 9
		bar = progressbar.ProgressBar(maxval=n1, widgets=[progressbar.Bar(marker='=', left='[', right=']'), ' ', progressbar.SimpleProgress()])

		bar.start()
		## Coloring the regions
		for i, region in enumerate(regions):
			bar.update(i + 1)
			self.colorRegion(res, region)

			if self.getRegionArea(region) > self.grayImg.shape[0] * self.grayImg.shape[1] * AREA_LIM:
				n2 += 1
				self.colorRegion(res2, region)

				if self.getRegionPerimeter(region) > 2 * (self.grayImg.shape[0] + self.grayImg.shape[1]) * PERIMETER_LIM:
					n3 += 1
					self.colorRegion(res3, region)

					if self.getAspectRatio(region) < ASPECT_RATIO_LIM:
						n4 += 1
						self.colorRegion(res4, region)

						if (self.getOccupyRate(region) > OCCUPATION_LIM[0]) and (self.getOccupyRate(region) < OCCUPATION_LIM[1]):
							n5 += 1
							self.colorRegion(res5, region)

							if (self.getCompactness(region) > COMPACTNESS_LIM[0]) and (self.getCompactness(region) < COMPACTNESS_LIM[1]):
								n6 += 1
								self.colorRegion(res6, region)

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
										# strokes = np.where(strokes == 255, 0, strokes)
									# else:
										# strokes = np.where(strokes == 120, 0, strokes)
								# cv2.imwrite("strokes/{}_{}.jpg".format(mean, std), strokes)

								if len(strokeWidths) > SWT_TOTAL_COUNT:
									n7 += 1
									self.colorRegion(res7, region)
									if std < SWT_STD_LIM:
										n8 += 1
										self.colorRegion(res8, region)

										strokeWidthSizeRatio = strokeWidth / (1.0 * max(self.getRegionShape(region)))
										if strokeWidthSizeRatio > STROKE_WIDTH_SIZE_RATIO_LIM:
											n9 += 1
											self.colorRegion(res9, region)

											strokeWidthVarianceRatio = (1.0 * strokeWidth) / (std ** std)
											if strokeWidthVarianceRatio > STROKE_WIDTH_VARIANCE_RATIO_LIM:
												n10 += 1
												res10 = self.colorRegion(res10, region)

		# pltShow((self.img, "Original"), (res, "Original image"))
		# plt.imsave("mser.jpg", res)
		# sys.exit()

		bar.finish()
		print("{} regions left.".format(n10))

		## Binarize regions
		binarized = np.zeros_like(self.grayImg)
		rows, cols, color = np.where(res10 != [0, 0, 0])
		binarized[rows, cols] = 255

		## Dilate regions and find contours
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
		kernel = np.zeros((KSIZE, KSIZE), dtype=np.uint8)
		kernel[(KSIZE // 2)] = 1

		dilated = cv2.dilate(binarized.copy(), kernel, iterations=ITERATION)
		image, contours, hierarchies = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if TESS:
			print("Tesseract eliminates..")

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
				else:
					rect = cv2.minAreaRect(contour)
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)

				# cv2.rectangle(self.final, (x, y), (x + w, y + h), (0, 255, 0), 2)

		pltShow((self.img, "Image"), \
				(self.cannyImg, "Canny"), \
				(res, "MSER,({} regions)".format(n1)), \
				(res2, "Min Area={},({} regions)".format(AREA_LIM, n2)), \
				(res3, "Min Perimeter={},({} regions)".format(PERIMETER_LIM, n3)), \
				(res4, "Aspect Ratio={},({} regions)".format(ASPECT_RATIO_LIM, n4)), \
				(res5, "Occupation={},({} regions)".format(OCCUPATION_LIM, n5)), \
				(res6, "Compactness={},({} regions)".format(COMPACTNESS_LIM, n6))
				)

		pltShow(# (strokes, "strokes"), \
				(res7, "STROKES TOTAL COUNT={},({} regions)".format(SWT_TOTAL_COUNT, n7)),\
				(res8, "STROKES STD={},({} regions)".format(SWT_STD_LIM, n8)), \
				(res9, "STROKE/SIZE RATIO={},({} regions)".format(STROKE_WIDTH_SIZE_RATIO_LIM, n9)),\
				(res10, "STROKE/VARIANCE RATIO={},({} regions)".format(STROKE_WIDTH_VARIANCE_RATIO_LIM, n10)),\
				(binarized, "Binarized"),\
				# (rotated, "Rotated"),\
				(dilated, "Dilated (iterations={},ksize={})".format(ITERATION, KSIZE)),\
				(self.final, "Final")\
				)

td = TextDetection(rgbImg)
td.detect()
# pltShow((td.img, "Original"), (td.final, "Final"))

plt.imshow(td.final)
plt.xticks([])
plt.yticks([])
plt.show()
