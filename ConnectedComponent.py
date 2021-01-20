
from Preprocessing import *
import cv2
import numpy as np
import math
from imutils.contours import sort_contours
import argparse
import imutils

# path = "aa"

def ConnectedComponents (img):
    backtorgb = img.copy()
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(backtorgb, (5, 5), 0)
    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    backtorgb = cv2.cvtColor(edged.copy(),cv2.COLOR_GRAY2RGB)
    img2 =backtorgb.copy()
    bounding_rect = np.zeros((len(contours), 5))


    for i in range(0, len(contours )):
            # imm = img2.copy()
            x, y, w, h = cv2.boundingRect(contours[i])
            if ((w == img2.shape[1]) or (w <= img2.shape[1]+10 and img2.shape[1]-10<=w)) or w<=5:
                continue 
            # cv2.rectangle(img2, (x,y), (x + w, y + h), (0,255,0), 1)
            # cv2.rectangle(imm, (x,y), (x + w, y + h), (0,255,0), 1)
            bounding_rect[i] = (int(x), int(y), int(w), int(h), float(h / w))   
 

    bounding_rect = bounding_rect[~np.all(bounding_rect == 0, axis=1)]

    
    #getting aspect ratio
    h_to_w_ratio = np.average(bounding_rect[:, 4], axis=0)
    #sort contrours based on top left 
    bounding_rect_sorted = bounding_rect[bounding_rect[:, 0].argsort()]
  
    #distance between each bounding box - width of the first bounding bx to get the distance between two bounding boxes
    diff_dist_word = np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2])
    threshold = np.average(np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]))
    word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])

    within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))
    #  if line consists of only one word
    if math.isnan(word_dist):
        word_dist = 0
    if math.isnan(within_word_dist):
        within_word_dist = 0

    sdW = np.std(bounding_rect_sorted[:, 2]) # varies in a specific range
    MedianW = np.median(bounding_rect_sorted[:, 2]) # in the middle of all numbers
    AverageW = np.average(bounding_rect_sorted[:, 2]) #mean
            
    return np.asarray([word_dist, within_word_dist, sdW, MedianW, AverageW, h_to_w_ratio])
