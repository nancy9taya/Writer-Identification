
import cv2 
import numpy as np

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