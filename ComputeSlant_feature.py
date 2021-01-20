
import skimage.filters as filters
import numpy as np
import math

"""
This Feature will be responsible to know the slant that the writer writes with
"""

"""
Inputs:
    image : 2D array of pixels for a binarized image(0 255)
Output:
    angleHist: distribution for edge angle 
    if the writer writes titled words the distribution will have 
    higher values at the small edge angles
p.s:
    the edge angle is the angle between edge fragment and the horizontal
"""
def AnglesHistogram(image):
    # print("Compute Slant feature...................\n")
    # print(image)
    _,count = np.unique(image, return_counts=True) 
    countBlack=count[0]
    totalEdges=np.sum(count)

    """
    sobel applies kernel on the image then calculate a gradient in horizontal direction and vertical direction
    getting the angle between thos 2 directions will yield the edge angle.
    """

    sob_img_v = filters.sobel_v(image)
    sob_img_h = filters.sobel_h(image)
    angles = np.arctan2(sob_img_v, sob_img_h)
    angles = np.multiply(angles, (180 / math.pi))
    angles = np.round(angles)
    anglesDist = [] # contains a distribution for angles for every interval between -180 to 180
                    # will correspond to number of angles that exist in this interval
    # print("ANGLESSSSSSSSSSS")
    # print(angles)
    anglesHist=[]
    start_angle = 10
    interval =30
    end_angle=start_angle+interval
    closed_interval=False

    while end_angle < 180:
        anglesCopy = angles.copy()
        anglesCopy[np.logical_or(anglesCopy < start_angle, anglesCopy > end_angle)] = 0
        if closed_interval==True:
            anglesCopy[np.logical_and(anglesCopy >= start_angle, anglesCopy <= end_angle)] = 1
        else:
            anglesCopy[np.logical_and(anglesCopy > start_angle, anglesCopy < end_angle)] = 1
        anglesHist.append(np.sum(anglesCopy))
        start_angle +=interval
        end_angle += interval
        closed_interval=not(closed_interval)

    # print(np.divide(anglesHist, countBlack))
    return np.divide(anglesHist, countBlack)