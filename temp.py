
from Preprocessing import *
import cv2
import numpy as np
import math
from commonfunctions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
import scipy.misc as misc

# path = "training (1).png"
# Binary=  Noise_Removal(path)
# cropedImage = cropImage(Binary)
# array = getLines(cropedImage)
# print("********************")
# print(array[0].shape)

# path = "training (1).png"
# Binary=  Noise_Removal(path)
# cropedImage = cropImage(Binary)
# array = getLines(cropedImage)



im = cv2.imread('arwa.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray, (5, 5), 0)
ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inverse = cv2.bitwise_not(imgray)
contours, hierarchy = cv2.findContours(thresh_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
backtorgb = cv2.cvtColor(thresh_inverse,cv2.COLOR_GRAY2RGB)
print(len(contours))
bounding_rect = np.zeros((len(contours), 6))
for i in range(0, len(contours)):
        img2 =backtorgb.copy()
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(backtorgb, (x,y), (x + w, y + h), (10,230,0), 10)
        #cv2.rectangle(img2, (x,y), (x + w, y + h), (0,255,0), 20)
        bounding_rect[i] = (int(x), int(y), int(w), int(h), int(w * h), int(h / w))
        # print("--------------------------------")
        # print(i,hnew[i],w,h)
        stre= str(i)
        #cv2.imwrite(stre+'.png', img2)
        #cv2.imshow('lll', img2)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
#cv2.drawContours(example_copy, contours, 2, (0, 230, 255), 6)

#getting aspect ratio
h_to_w_ratio = np.average(bounding_rect[:, 5], axis=0)

#sort contrours based on top left 
bounding_rect_sorted = bounding_rect[bounding_rect[:, 0].argsort()]

#distance between each bounding box - width of the first bounding bx to get the distance between two bounding boxes
diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
threshold = np.average(np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]))
word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])
within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))

sdW = np.sqrt(np.var(bounding_rect_sorted[:, 2])) # varies in a specific range
MedianW = np.median(bounding_rect_sorted[:, 2]) # in the middle of all numbers
AverageW = np.average(bounding_rect_sorted[:, 2]) #mean

print(word_dist, within_word_dist, sdW, MedianW, AverageW, h_to_w_ratio)

# if line consists of only one word
# if math.isnan(word_dist):
#     word_dist = 0
# if math.isnan(within_word_dist):
#     within_word_dist = 0
        

cv2.imwrite('arwap.png', backtorgb)
cv2.waitKey()
cv2.destroyAllWindows()
