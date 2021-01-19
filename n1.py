
from Preprocessing import *
import cv2
import numpy as np
import math
img = cv2.imread("outputs\line1.png",0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  
#ret, labels = cv2.connectedComponents(img)
#ret, labels = cv2.connectedComponents(img)
def remove_small_objects(img, min_size=150):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # your answer image
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 0

        return img2 
im = remove_small_objects(img.copy())    
# cv2.imshow('image2',im)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, 150, 3)
for i in range(0, len(contours )):
    x, y, w, h = cv2.boundingRect(contours[i])
    # if ((w == im.shape[1]) or (w+10>im.shape[1] and im.shape[1]<w-10)) :
    #     continue 
    cv2.rectangle(img, (x,y), (x + w, y + h), 30, 1)
    print(w)
cv2.imwrite('arwap.png', img)
print("shape",im.shape)
cv2.waitKey()
cv2.destroyAllWindows()