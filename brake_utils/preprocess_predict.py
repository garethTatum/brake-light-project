# Simple mask test

import numpy as np
from skimage import measure
from imutils import contours
import argparse
import imutils
import cv2
import time
from brake_utils.random_forest_manual_train import RandomForest
font = cv2.FONT_HERSHEY_SIMPLEX

def adjust_gamma(img, gamma=1):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def lab(img):
        p=0.08
        img_orig = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        lower_value1 = np.array([84,184,175])
        lower_value1 = lower_value1 - (lower_value1 * p)
        upper_value1 = np.array([137,208,195])
        upper_value1 = upper_value1 + (upper_value1 * p)

        lower_value2 = np.array([196,106,152])
        lower_value2 = lower_value2 - (lower_value2 * p)
        upper_value2 = np.array([252,149,223])
        upper_value2 = upper_value2 + (upper_value2 * p)

        threshold1 = cv2.inRange(img, lower_value1, upper_value1)
        threshold1 = cv2.erode(threshold1, None, iterations=0)
        threshold1 = cv2.dilate(threshold1, None, iterations=0)

        threshold2 = cv2.inRange(img, lower_value2, upper_value2)
        threshold2 = cv2.erode(threshold2, None, iterations=0)
        threshold2 = cv2.dilate(threshold2, None, iterations=0)

        restogether_mask = cv2.add(threshold1, threshold2)

        final_res = cv2.bitwise_and(img_orig,img_orig, mask = restogether_mask)

        return final_res

def brakecheck4(image, rf):
    img = image
    img = lab(img)
    result = rf.predict(img)

    writetoimg = str(result)
    cv2.putText(image, writetoimg, (0, 20), cv2.FONT_HERSHEY_SIMPLEX ,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if result > 0.60:
        return 1
    else:
        return 0
