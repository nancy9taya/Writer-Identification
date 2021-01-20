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
import csv
import threading
import time

testDataPath="TestData"  ## Change this to be path of folder of test data set

num_Iterations=0
Y_train1=[]
Y_train2=[]
Y_train3=[]

X_test=[]
Y_test=[]

# X_Train=[]
# Y_Train=[]
times=[]
preds=[]

Cnt_segmentedLines=[]
# img_writer=dict()
writer_1="1"
writer_2="2"
writer_3="3"



# prepare X (Feature vector) and Y(the right answer) 
def processImages(X,Y,y,imgPath):
    segmentedLines=preprocessImage(imgPath)
    for line in segmentedLines:
        if len(line)==0:
            continue
        r=getFeatureVector(line.copy())
        if len(r)!=0:
            Y.append(y)
            X.append(r)

def processTestImage(imgPath):
    segmentedLines=preprocessImage(imgPath)
    # print(segmentedLines)
    for line in segmentedLines:
        X_test.append(getFeatureVector(line))
    return X_test

def writePrediction(pred,time):
    fr.write(str(pred))
    fr.write("\n")
    time = str(round(time, 2))
    ft.write(str(time))
    ft.write("\n")


def ReadData(testDataPath,num_Iterations):
    iteration_Folders=os.listdir(testDataPath)
    global X_test
    for i in range(num_Iterations):
        folder=iteration_Folders[i]
        start = time.time()
        X_Train=[]
        Y_Train=[]
        X_test=[]
        X_train1=[]
        X_train2=[]
        Y_train1=[]
        Y_train2=[]
        print("Test Case ..............", i+1)
        writers_folder=os.listdir(str(testDataPath+"/"+folder))
        if len(writers_folder)==0:
            continue;
        for i in range(4): # loop over writers folders and test image
            writer_folder=writers_folder[i]
            if i==3:
                testPath=str(testDataPath+"/"+folder+"/"+writer_folder)
                processTestImage(testPath)
                continue
            imgs=os.listdir(str(testDataPath+"/"+folder+"/"+writer_folder))
            t1=None
            t2=None
            for idx,img in enumerate(imgs):
                if idx==0:
                    imgPath=str(testDataPath+"/"+folder+"/"+writer_folder+"/"+img)
                    t1 = threading.Thread(target=processImages, args=(X_train1,Y_train1,i+1,imgPath)) 
                    t1.start()
                if idx==1:
                    imgPath=str(testDataPath+"/"+folder+"/"+writer_folder+"/"+img)
                    # processImages(img,X_train2,Y_train2,i+1,imgPath)
                    t2 =threading.Thread(target=processImages, args=(X_train2,Y_train2,i+1,imgPath)) 
                    t2.start()
            t1.join()
            # print("t1",t1.is_alive())
            t2.join()
            # print("t2",t1.is_alive())
            print("Extracting features.....................")
            X_Train.extend(X_train1)
            X_Train.extend(X_train2)
            Y_Train.extend(Y_train1)
            Y_Train.extend(Y_train2)
        print("Fitting classifier..................")
        training(X_Train,Y_Train)
        print("PREDICTION BEGINS NOW ! ...............")
        # print(X_test)
        pred=predict_clf(X_test,Y_test)
        end=time.time()
        print("Time Ellapsed ", end-start," seconds")
        times.append(end-start)
        writePrediction(pred,end-start)
        print(pred)


# ResultsFolder="Results"
if os.path.exists(str(testDataPath+"/results.txt")):
    os.remove(str(testDataPath+"/results.txt"))
if os.path.exists(str(testDataPath+"/time.txt")):
    os.remove(str(testDataPath+"/time.txt"))
num_Iterations=len(os.listdir(testDataPath))
ft = open(testDataPath+"/time.txt", "a")
fr = open(testDataPath+"/results.txt", "a")
ReadData(testDataPath,num_Iterations)
ft.close()
fr.close()