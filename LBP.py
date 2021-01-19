from skimage import feature
import numpy as np
import sys
import cv2 
import matplotlib.pyplot as plt

def extractContours(line):
    image = line
    blur = cv2.GaussianBlur(image, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    extractedImages = []
    padding = 2
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if(h*w) > 50  :
            xmin = x+padding
            xmax = x + w +padding
            ymin = y  +padding
            ymax = y + h +padding
            extractedImages.append(image[ymin: ymax,xmin:xmax])  

    return extractedImages

def LBP(image, eps=1e-7):
    lbp = feature.local_binary_pattern(image, 8,1, method="uniform")
    n_bins= int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp.ravel(),bins=n_bins,range=(0, n_bins))
    #plt.hist(lbp.ravel(),bins=n_bins,range=(0, n_bins))
    # normalize the histogram
    print(hist)
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist

def applyLBP(extractedImages):
    featuers = []
    for i in range(len(extractedImages)):  
        featuers.append(LBP(extractedImages[i]).tolist())
    return featuers