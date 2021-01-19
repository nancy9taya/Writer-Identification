from skimage import feature
import numpy as np
import sys
import cv2 
import matplotlib.pyplot as plt


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