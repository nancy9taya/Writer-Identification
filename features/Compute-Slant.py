import skimage.filters as filters
import numpy as np
import opencv as cv2
from skimage.filters import sobel_h, sobel_v

def eightDirections(window):
    x = 2
    y = 2
    windowHist = np.zeros((1, 9))
    if window[x, y] == 0:
        return windowHist
    windowHist[0][0] = window[x + 1, y] & window[x + 2, y]
    windowHist[0][1] = window[x + 1, y - 1] & window[x + 2, y - 1]
    windowHist[0][2] = window[x + 1, y - 1] & window[x + 2, y - 2]
    windowHist[0][3] = window[x, y - 1] & window[x + 1, y - 2]
    windowHist[0][4] = window[x, y - 1] & window[x, y - 2]
    windowHist[0][5] = window[x, y - 1] & window[x - 1, y - 2]
    windowHist[0][6] = window[x - 1, y - 1] & window[x - 2, y - 2]
    windowHist[0][7] = window[x - 1, y - 1] & window[x - 2, y - 1]
    windowHist[0][8] = window[x - 1, y] & window[x - 2, y]
    return windowHist


def computeSlantHistogram(line):
    line = np.array(line)
    scale_percent = 25
    width = int(line.shape[1] * scale_percent / 100)
    height = int(line.shape[0] * scale_percent / 100)
    dim = (width, height)
    lineResize = cv2.resize(line, dim, interpolation=cv2.INTER_AREA)
    edgeX = sobel_h(lineResize)
    edgeY = sobel_v(lineResize)
    histogram = np.zeros((1, 9))
    h, w = lineResize.shape
    lineResize[lineResize == 255] = 1
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            window = lineResize[i - 2:i + 3, j - 2:j + 3]
            windowHistogram = eightDirections(window)
            histogram = histogram + windowHistogram
    sumHistogram=np.sum(histogram)
    if sumHistogram!=0:
        histogram = histogram /sumHistogram
    return histogram

    