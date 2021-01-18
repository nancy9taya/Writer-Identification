from Preprocessing import *
from classification import *
from FeatureExtraction import *

import skimage.io as io
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import cv2
import os

X_Train=[]
X_Test=[]
Y_Train=[]
Y_Test=[]

def processImages(trainingFolder):
    # writerArr=list(range(numWriters))
    #read hand writing images from dataset training folder
    file_list=os.listdir(trainingFolder) # get files name
    print("in process Training iamges function.....................................")
    print(len(file_list))
    # for file in file_list:
    #     segmentedLines=preprocessImage(trainingFolder+"/"+file)
    #     for line in segmentedLines:
    #         X_Train.extend(getFeatureVector(line)) # X_Train now is array of Segmented Lines
    return X_Train







# def main():
"""
formsA-D 529
formsE-H 395
formsl-Z 458
forms.txt 
"""
trainingDataPath="formsI-Z" 
# trainingDataPath="IAmData"
testDataPath="TestData"

"""
# Transfrm ASCII txt file to CSV
# df = pd.read_csv('ascii/forms.csv',sep='\s+',header=None)
# df.to_csv('ascii/form_out.csv',header=None)
"""
Y_Train=pd.read_csv(filepath_or_buffer='ascii/form_out.csv', header=None, usecols=[1,2]);
# data = pd.read_csv("data.csv")
Y_train = {col: list(Y_Train[col]) for col in Y_Train.columns}
# Y_Train=dict(Y_Train)
# print(Y_Train['a01-000x'])
# first_column = Y_Train.columns[0]
# Delete first
# Y_Train = Y_Train.drop([first_column], axis=1)
# Y_Train.to_csv('file.csv', index=False)
print(Y_Train)
print("Processsing the images..............")
# X_Train is array of feature vectors for every line in images of training data set
# X_Train is 2-D np.array
# Y_Train is 1-D array contains correct classification.
X_train=processImages(trainingDataPath)
# training(X_Train,Y_Train)
# Y_Train=processImages(testDataPath)


