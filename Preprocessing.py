from commonfunctions import *
import numpy as np
import sys
import cv2 
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import  binary_erosion, binary_dilation, binary_closing,skeletonize, thin,square,binary_opening,disk
from skimage.measure import find_contours
from skimage.draw import rectangle
from copy import deepcopy
from skimage.transform import resize
from skimage.filters import median, gaussian



def Noise_Removal(path):
    image = io.imread(path, as_gray=True)
    med_img = median(image, disk(1), mode='constant', cval=0.0)
    gray = rgb2gray(med_img)
    threshold = threshold_otsu(gray)  
    Binary = gray > threshold 
    Binary = (Binary*255).astype('uint8')
    return Binary






def cropImage(Binary):
    scale= 900/Binary.shape[0]
    resized_img = resize(Binary, ( int(Binary.shape[0] *scale), int(Binary.shape[1] * scale)))
    resized_img  = (resized_img*255).astype('uint8') 
    thresh = 255 -resized_img
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)# apply openenig
    edges = cv2.Canny(detected_lines,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(image=edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)
    a,b,c = lines.shape
    vertices = [ ]
    for i in range(a):
        vertices.append(lines[i][0][1])

    vertices = sorted(vertices)
    vertices =np.array(vertices)
    dif = vertices[1:] - vertices[:-1]
    # get the index where you first observe a jump
    fi =  np.where(abs(dif) > 5)
    out = [item for t in fi for item in t] 
    uniques=[]
    uniques= [vertices[i+1] for i in out ]
    uniques.insert(0,vertices[0])
    uniques = sorted(uniques)
    cropedImage = resized_img[uniques[1]+5:uniques[2]-5, :]
    cv2.imwrite('croped_image.png', cropedImage)
    return cropedImage

def RemoveBLackEdge(cropedImage):
    img = cropedImage.T
    w,h=img.shape
    sens=1.0 # (0-1]
    meanofimg=np.mean(img)*sens
    dataw=[w,0]
    datah=[h,0]
    for i in range(w):
        if np.mean(img[i])>meanofimg:
            if i<dataw[0]:
                dataw[0]=i
            else:
                dataw[1]=i
    img=img.T
    meanofimg=np.mean(img)*sens
    for i in range(h):
        if np.mean(img[i])>meanofimg:
            if i<datah[0]:
                datah[0]=i
            else:
                datah[1]=i
    img=img[datah[0]:datah[1],dataw[0]:dataw[1]]
    return img



 
def getLines(cropedImage):   
    linesArray = []
    th, threshed = cv2.threshold(cropedImage, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    th = 2
    H,W = cropedImage.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    Neg = cropedImage
    if(len(uppers) == len(lowers)):
        for i in range(len(uppers)):
            if(abs(uppers[i] -lowers[i]) >= 10):
                linesArray.append(Neg [uppers[i]+2:lowers[i]+2,:])
                line = Neg [uppers[i]+2:lowers[i]+2,:]
                cv2.imwrite("outputs/line"+str(i)+".jpg",line)
    elif(len(uppers) > len(lowers)):
        for i in range(len(lowers)):
            if(abs(uppers[i] -lowers[i]) >= 10):
                linesArray.append(Neg [uppers[i]+2:lowers[i]+2,:])
                line = Neg [uppers[i]+2:lowers[i]+2,:]
                cv2.imwrite("outputs/line"+str(i)+".jpg",line)
    elif(len(uppers) < len(lowers)):
        for i in range(len(uppers)):
            if(abs(uppers[i] -lowers[i]) >= 10):
                linesArray.append(Neg [uppers[i]+2:lowers[i]+2,:])
                line = Neg [uppers[i]+2:lowers[i]+2,:]
                cv2.imwrite("outputs/line"+str(i)+".jpg",line)

    #just for visualizing
    for y in uppers:
        cv2.line(Neg , (0,y), (W, y), (255,0,0), 1)

    for y in lowers:
        cv2.line(Neg , (0,y), (W, y), (0,255,0), 1)

    cv2.imwrite("outputs/result.png", Neg )
    return linesArray



"""
    Preprocessing Steps
"""
def preprocessImage(path):
    binImage=Noise_Removal(path)
    croppedImage=cropImage(binImage)
    fineImage = RemoveBLackEdge(croppedImage)
    segmentedLines=getLines(fineImage)
    return  segmentedLines
