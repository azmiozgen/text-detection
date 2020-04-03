import os

import cv2
import numpy as np
from scipy.stats import mode, norm
from pytesseract import image_to_string, image_to_boxes
from PIL import Image

from progressbar import ProgressBar, Bar, SimpleProgress

from lib.region import Region
from lib.utils import plt_show, apply_canny


class TextDetection(object):

    def __init__(self, image_file, config, direction='both+', use_tesseract=True, details=False):

        ## Read image
        self.image_file = image_file
        img = cv2.imread(image_file)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = rgb_img
        self.h, self.w = img.shape[:2]

        self.direction = direction
        self.use_tesseract = use_tesseract
        self.details = details
        self.config = config
        self.AREA_LIM = config.AREA_LIM
        self.PERIMETER_LIM = config.PERIMETER_LIM
        self.ASPECT_RATIO_LIM = config.ASPECT_RATIO_LIM
        self.OCCUPATION_INTERVAL = config.OCCUPATION_INTERVAL
        self.COMPACTNESS_INTERVAL = config.COMPACTNESS_INTERVAL
        self.SWT_TOTAL_COUNT = config.SWT_TOTAL_COUNT
        self.SWT_STD_LIM = config.SWT_STD_LIM
        self.STROKE_WIDTH_SIZE_RATIO_LIM = config.STROKE_WIDTH_SIZE_RATIO_LIM
        self.STROKE_WIDTH_VARIANCE_RATIO_LIM = config.STROKE_WIDTH_VARIANCE_RATIO_LIM
        self.STEP_LIMIT = config.STEP_LIMIT
        self.KSIZE = config.KSIZE
        self.ITERATION = config.ITERATION
        self.MARGIN = config.MARGIN

        self.final = rgb_img.copy()

        self.height, self.width = self.img.shape[:2]

        self.gray_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)

        self.canny_img = apply_canny(self.img)

        self.sobelX = cv2.Sobel(self.gray_img, cv2.CV_64F, 1, 0, ksize=-1)
        self.sobelY = cv2.Sobel(self.gray_img, cv2.CV_64F, 0, 1, ksize=-1)

        self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
        self.stepsY = self.sobelX.astype(int)

        self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
        self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
        self.gradsY = self.stepsY / (self.magnitudes + 1e-10)

    def get_MSERegions(self, img):
        mser = cv2.MSER_create()
        regions, bboxes = mser.detectRegions(img)
        return regions, bboxes

    def get_stroke_properties(self, stroke_widths):
        if len(stroke_widths) == 0:
            return (0, 0, 0, 0, 0, 0)
        try:
            most_probable_stroke_width = mode(stroke_widths, axis=None)[0][0]
            most_probable_stroke_width_count = mode(stroke_widths, axis=None)[1][0]
        except IndexError:
            most_probable_stroke_width = 0
            most_probable_stroke_width_count = 0
        try:
            mean, std = norm.fit(stroke_widths)
            x_min, x_max = int(min(stroke_widths)), int(max(stroke_widths))
        except ValueError:
            mean, std, x_min, x_max = 0, 0, 0, 0
        return most_probable_stroke_width, most_probable_stroke_width_count, mean, std, x_min, x_max

    def get_strokes(self, xywh):

        x, y, w, h = xywh
        stroke_widths = np.array([[np.Infinity, np.Infinity]])
        for i in range(y, y + h):
            for j in range(x, x + w):
                if self.canny_img[i, j] != 0:
                    gradX = self.gradsX[i, j]
                    gradY = self.gradsY[i, j]

                    prevX, prevY, prevX_opp, prevY_opp, step_size = i, j, i, j, 0

                    if self.direction == "light":
                        go, go_opp = True, False
                    elif self.direction == "dark":
                        go, go_opp = False, True
                    else:
                        go, go_opp = True, True

                    stroke_width = np.Infinity
                    stroke_width_opp = np.Infinity
                    while (go or go_opp) and (step_size < self.STEP_LIMIT):
                        step_size += 1

                        if go:
                            curX = np.int(np.floor(i + gradX * step_size))
                            curY = np.int(np.floor(j + gradY * step_size))
                            if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
                                go = False
                            if go and ((curX != prevX) or (curY != prevY)):
                                try:
                                    if self.canny_img[curX, curY] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[curX, curY]) < np.pi / 2.0:
                                            stroke_width = int(np.sqrt((curX - i) ** 2  + (curY - j) ** 2))
                                            go = False
                                except IndexError:
                                    go = False

                                prevX = curX
                                prevY = curY

                        if go_opp:
                            curX_opp = np.int(np.floor(i - gradX * step_size))
                            curY_opp = np.int(np.floor(j - gradY * step_size))
                            if (curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w):
                                go_opp = False
                            if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                                try:
                                    if self.canny_img[curX_opp, curY_opp] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[curX_opp, curY_opp]) < np.pi/2.0:
                                            stroke_width_opp = int(np.sqrt((curX_opp - i) ** 2  + (curY_opp - j) ** 2))
                                            go_opp = False

                                except IndexError:
                                    go_opp = False

                                prevX_opp = curX_opp
                                prevY_opp = curY_opp

                    stroke_widths = np.append(stroke_widths, [(stroke_width, stroke_width_opp)], axis=0)

        stroke_widths_opp = np.delete(stroke_widths[:, 1], np.where(stroke_widths[:, 1] == np.Infinity))
        stroke_widths = np.delete(stroke_widths[:, 0], np.where(stroke_widths[:, 0] == np.Infinity))
        return stroke_widths, stroke_widths_opp

    def detect(self):
        res9 = np.zeros_like(self.img)
        if self.details:
            res0 ,res1, res2, res3, res4, res5, res6, res7, res8 = res9.copy(), res9.copy(), res9.copy(), \
                                                                   res9.copy(), res9.copy(), res9.copy(), \
                                                                   res9.copy(), res9.copy(), res9.copy()
        regions, bboxes = self.get_MSERegions(self.gray_img)
        #TODO regions, bboxes = self.get_MSERegions(self.img)

        n_mser_regions = len(regions)
        n_final_regions = 0
        if self.details:
            n1, n2, n3, n4, n5, n6, n7, n8, n9 = [0] * 9

        bar = ProgressBar(maxval=n_mser_regions, widgets=[Bar(marker='=', left='[', right=']'), ' ', SimpleProgress()])
        bar.start()

        for i, (region, bbox) in enumerate(zip(regions, bboxes)):
            bar.update(i + 1)

            region = Region(region, bbox)
            if self.details:
                res0 = region.color(res0)

            if region.area < self.w * self.h * self.AREA_LIM:
                continue
            if self.details:
                res1 = region.color(res1)
                n1 += 1

            if region.get_perimeter(self.canny_img) < (2 * (self.w + self.h) * self.PERIMETER_LIM):
                continue
            if self.details:
                res2 = region.color(res2)
                n2 += 1

            if region.get_aspect_ratio() > self.ASPECT_RATIO_LIM:
                continue
            if self.details:
                res3 = region.color(res3)
                n3 += 1

            occupation = region.get_occupation()
            if (occupation < self.OCCUPATION_INTERVAL[0]) or (occupation > self.OCCUPATION_INTERVAL[1]):
                continue
            if self.details:
                res4 = region.color(res4)
                n4 += 1

            compactness = region.get_compactness()
            if (compactness < self.COMPACTNESS_INTERVAL[0]) or (compactness > self.COMPACTNESS_INTERVAL[1]):
                continue
            if self.details:
                res5 = region.color(res5)
                n5 += 1

            x, y, w, h = bbox

            stroke_widths, stroke_widths_opp = self.get_strokes((x, y, w, h))
            if self.direction != "both+":
                stroke_widths = np.append(stroke_widths, stroke_widths_opp, axis=0)
                stroke_width, stroke_width_count, _, std, _, _ = self.get_stroke_properties(stroke_widths)
            else:
                stroke_width, stroke_width_count, _, std, _, _ = self.get_stroke_properties(stroke_widths)
                stroke_width_opp, stroke_width_count_opp, _, std_opp, _, _ = self.get_stroke_properties(stroke_widths_opp)
                if stroke_width_count_opp > stroke_width_count:        ## Take the stroke_widths with max of counts stroke_width (most probable one)
                    stroke_widths = stroke_widths_opp
                    stroke_width = stroke_width_opp
                    stroke_width_count = stroke_width_count_opp
                    std = std_opp

            if len(stroke_widths) < self.SWT_TOTAL_COUNT:
                continue
            if self.details:
                res6 = region.color(res6)
                n6 += 1

            if std > self.SWT_STD_LIM:
                continue
            if self.details:
                res7 = region.color(res7)
                n7 += 1

            stroke_width_size_ratio = stroke_width / max(region.w, region.h)
            if stroke_width_size_ratio < self.STROKE_WIDTH_SIZE_RATIO_LIM:
                continue
            if self.details:
                res8 = region.color(res8)
                n8 += 1

            stroke_width_variance_ratio = stroke_width / (std * std + 1e-10)
            if stroke_width_variance_ratio > self.STROKE_WIDTH_VARIANCE_RATIO_LIM:
                n_final_regions += 1
                res9 = region.color(res9)
                if self.details:
                    n9 += 1

        bar.finish()
        print("{} regions left.".format(n_final_regions))

        ## Binarize regions
        binarized = np.zeros_like(self.gray_img)
        rows, cols, _ = np.where(res9 != [0, 0, 0])
        binarized[rows, cols] = 255

        ## Dilate regions and find contours
        kernel = np.zeros((self.KSIZE, self.KSIZE), dtype=np.uint8)
        kernel[(self.KSIZE // 2)] = 1

        res = np.zeros_like(self.gray_img)
        dilated = cv2.dilate(binarized.copy(), kernel, iterations=self.ITERATION)
        _, contours, hierarchies = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.use_tesseract:
            print("Tesseract will eliminate..")
            temp_file = 'text.jpg'
        for i, (contour, hierarchy) in enumerate(zip(contours, hierarchies[0])):
            if hierarchy[-1] != -1:
                continue

            if self.use_tesseract:
                x, y, w, h = cv2.boundingRect(contour)
                if (y - self.MARGIN > 0) and (y + h + self.MARGIN < self.height) and (x - self.MARGIN > 0) and (x + w + self.MARGIN < self.width):
                    cv2.imwrite(temp_file, self.final[y - self.MARGIN:y + h + self.MARGIN, x - self.MARGIN:x + w + self.MARGIN])
                else:
                    cv2.imwrite(temp_file, self.final[y:y + h, x:x + w])

                ## Run tesseract
                string = image_to_string(Image.open(temp_file))
                if string is not u'':
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
                    cv2.drawContours(res, [box], 0, 255, -1)
                os.remove(temp_file)

            else:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
                cv2.drawContours(res, [box], 0, 255, -1)

        if self.details:
            plt_show((self.img, "Image"), \
                     (self.canny_img, "Canny"), \
                     (res0, "MSER,({} regions)".format(n_mser_regions)), \
                     (res1, "Min Area={},({} regions)".format(self.AREA_LIM, n1)), \
                     (res2, "Min Perimeter={},({} regions)".format(self.PERIMETER_LIM, n2)), \
                     (res3, "Aspect Ratio={},({} regions)".format(self.ASPECT_RATIO_LIM, n3)), \
                     (res4, "Occupation={},({} regions)".format(self.OCCUPATION_INTERVAL, n4)), \
                     (res5, "Compactness={},({} regions)".format(self.COMPACTNESS_INTERVAL, n5))
                    )

            plt_show((res6, "STROKES TOTAL COUNT={},({} regions)".format(self.SWT_TOTAL_COUNT, n6)),\
                     (res7, "STROKES STD={},({} regions)".format(self.SWT_STD_LIM, n7)), \
                     (res8, "STROKE/SIZE RATIO={},({} regions)".format(self.STROKE_WIDTH_SIZE_RATIO_LIM, n8)),\
                     (res9, "STROKE/VARIANCE RATIO={},({} regions)".format(self.STROKE_WIDTH_VARIANCE_RATIO_LIM, n9)),\
                     (binarized, "Binarized"),\
                     (dilated, "Dilated (iterations={},ksize={})".format(self.ITERATION, self.KSIZE)),\
                     (self.final, "Final")\
                    )

        return res

    def full_OCR(self):
        bounded = self.img.copy()
        res = np.zeros_like(self.gray_img)

        string = image_to_string(Image.open(self.image_file))
        if string == u'':
            return bounded, res

        boxes = image_to_boxes(Image.open(self.image_file))
        boxes = [map(int, i) for i in [b.split(" ")[1:-1] for b in boxes.split("\n")]]

        for box in boxes:
            b = (int(box[0]), int(self.h - box[1]), int(box[2]), int(self.h - box[3]))
            cv2.rectangle(bounded, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.rectangle(res, (b[0], b[1]), (b[2], b[3]), 255, -1)

        return bounded, res
