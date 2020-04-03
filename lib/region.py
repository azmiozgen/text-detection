import cv2
import numpy as np

class Region(object):
    '''
    Takes Maximally Stable Extremal Region and computes a collection region properties
    '''

    def __init__(self, MSERegion, bbox):
        self.region = MSERegion
        # self.x, self.y, self.w, self.h = self._set_shape()
        self.x, self.y, self.w, self.h = bbox
        self.area = self._set_area()

        self.perimeter = None

    def _set_shape(self):
        x, y, w, h = cv2.boundingRect(self.region)
        return x, y, w, h
        # return max(self.region[:, 1]) - min(self.region[:, 1]), max(self.region[:, 0]) - min(self.region[:, 0])

    def _set_area(self):
        return len(list(self.region))        ## Number of pixels

    def get_perimeter(self, canny_img):
        self.perimeter = len(np.where(canny_img[self.y:self.y + self.h, self.x:self.x + self.w] != 0)[0])
        return self.perimeter
        #TODO perimeter = cv.arcLength(cnt,True)

    def get_occupation(self):
        return self.area / (self.w * self.h + 1e-10)

    def get_aspect_ratio(self):
        return max(self.w, self.h) / (min(self.w, self.h) + 1e-10)

    def get_compactness(self):
        if self.perimeter:
            return self.area / (self.perimeter ** 2 + 1e-10)
        else:
            return None

    def get_solidity(self):
        return self.area / (self.w * self.h + 1e-10)

    def color(self, img):
        img[self.region[:, 1], self.region[:, 0], 0] = np.random.randint(low=100, high=256)
        img[self.region[:, 1], self.region[:, 0], 1] = np.random.randint(low=100, high=256)
        img[self.region[:, 1], self.region[:, 0], 2] = np.random.randint(low=100, high=256)
        return img