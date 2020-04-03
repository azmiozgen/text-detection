import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

def plt_show(*images):
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

# def applyCloseEdgeDetect(img):
#     res = img.copy()
#     quantized = img
#     # quantized = applyKmeans(rgbImg, 5)
#     grayImg = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)

#     canny = cv2.Canny(grayImg, 120, 200)
#     # sobelX = cv2.Sobel(grayImg, cv2.CV_8U, 1, 0, ksize=3)
#     # sobelY = cv2.Sobel(grayImg, cv2.CV_8U, 0, 1, ksize=3)
#     # sobelX = cv2.convertScaleAbs(sobelX)
#     # sobelY = cv2.convertScaleAbs(sobelY)
#     # sobel = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0);

#     ret, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
#     morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     if cv2.__version__.startswith("2"):
#         contours, hierarchy = cv2.findContours(morphed, 0, 1)
#     else:
#         im, contours, hierarchy = cv2.findContours(morphed, 0, 1)
#     for contour in contours:
#         if len(contour) > 100:
#             contours_poly = cv2.approxPolyDP(contour, 3, True)
#             x, y, w, h = cv2.boundingRect(contours_poly)
#             cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     pltShow(img, quantized, canny, morphed, res)

# def applyGradientDetect(img):
#     res = img.copy()
#     quantized = img
#     # quantized = applyKmeans(rgbImg, 5)
#     grayImg = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     grad = cv2.morphologyEx(grayImg, cv2.MORPH_GRADIENT, kernel)

#     ret, thresh = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
#     connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     if cv2.__version__.startswith("2"):
#         contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     else:
#         im, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     for contour in contours:
#         contour_poly = cv2.approxPolyDP(contour, 3, True)
#         x, y, w, h = cv2.boundingRect(contour_poly)
#         cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     pltShow(img, quantized, grad, connected, res)

# def deskew(self, img):

#     if img.shape[-1] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     coords = np.column_stack(np.where(img > 0))
#     angle = cv2.minAreaRect(coords)[-1]

#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     # Rotate
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     return rotated, angle

# def rotateRect(self, x, y, w, h, angle):
#     newPoints = []
#     radius = math.sqrt((x - w / 2)**2 + (y - h / 2)**2)
#     center = (x + w / 2, y + h / 2)
#     points = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
#     for point in points:
#         angle += math.atan2(point[1] - center[1], point[0] - center[0])
#         newPoints.append( (round(center[0] + radius * math.cos(angle)), round(center[1] + radius * math.sin(angle))) )

#     return newPoints

# def applyKmeans(img, kVal, ATTEMPT=10):
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1.0)
#     data = img.reshape((-1, 3))
#     data = np.float32(data)
#     if cv2.__version__.startswith("2"):
#         compactness, labels, centers = cv2.kmeans(data, kVal, criteria, ATTEMPT, cv2.KMEANS_PP_CENTERS)
#     else:
#         compactness, labels, centers = cv2.kmeans(data, kVal, None, criteria, ATTEMPT, cv2.KMEANS_PP_CENTERS)
#     centers = np.uint8(centers)
#     kRes = centers[labels.flatten()]
#     kRes = kRes.reshape((img.shape))
#     return kRes