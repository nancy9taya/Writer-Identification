import numpy as np
import sys
import cv2 
from skimage.filters import gabor_kernel

#calculte different orienation of filters
def gabor_filter_bank():
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels     

# calculte features  [local energy,mean amplitude]        
def calculate_gabor_featuers(imageBinay ,kernels):
    featuers = []

    #calculating the local energy for each convolved image
    for i in range(len(kernels)):
        response = cv2.filter2D(imageBinay, cv2.CV_8UC3, kernels[i])#convolution
        squareElement = np.square(response)
        total =np.sum(squareElement)
        featuers.append(total )
    #calculating the mean amplitude for each convolved image
    for i in range(len(kernels)):
        response = cv2.filter2D(imageBinay, cv2.CV_8UC3, kernels[i])#convolution
        absElement = np.absolute(response)
        total =np.sum(absElement)
        featuers.append(total )
    return featuers

