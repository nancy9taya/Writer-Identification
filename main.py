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


X_train1=[]
X_train2=[]
X_train3=[]

Y_train1=[]
Y_train2=[]
Y_train3=[]

X_test=[]
Y_test=[]

X_Train=[]
Y_Train=[]
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
        Y.append(y)
        X.append(getFeatureVector(line)) 

def processTestImage(imgPath):
    segmentedLines=preprocessImage(imgPath)
    for line in segmentedLines:
        X_test.append(getFeatureVector(line))
    return X_test,len(segmentedLines)
"""
formsA-D 529
formsE-H 395
formsl-Z 458
forms.txt 
"""
testDataPath="TestData"

"""
# Transfrm ASCII txt file to CSV
# df = pd.read_csv('ascii/forms.csv',sep='\s+',header=None)
# df.to_csv('ascii/form_out.csv',header=None)
"""
def writePrediction(pred,time):
    fr.write(str(pred))
    fr.write("\n")
    ft.write(str(time))
    ft.write("\n")


def ReadData(testDataPath):
    global X_Train
    global Y_Train
    global X_test
    iteration_Folders=os.listdir(testDataPath)
    for idx,folder in enumerate(iteration_Folders):
        print("Test Case ..............", idx)
        writers_folder=os.listdir(str(testDataPath+"/"+folder))
        if len(writers_folder)==0:
            continue;
        if idx==5: ####################################to be removed 
            return
        # print(writers_folder)
        for i in range(4): # loop over writers folders and test image
            writer_folder=writers_folder[i]
            if i==3:
                testPath=str(testDataPath+"/"+folder+"/"+writer_folder)
                start = time.time()
                _,cntLines=processTestImage(testPath)
                Cnt_segmentedLines.append(cntLines)
                end=time.time()
                print("Time Ellapsed ", end-start," seconds")
                times.append(end-start)
                continue
            imgs=os.listdir(str(testDataPath+"/"+folder+"/"+writer_folder))
            t1=None
            t2=None
            for idx,img in enumerate(imgs):
                # print("Curreeeenttttt image",img)
                # print("IDXXXXXXXXXXXXXXXXXX",idx)
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
            t2.join()
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
        writePrediction(pred,end-start)
        print(pred)
        X_Train=[]
        Y_Train=[]
        X_test=[]


ResultsFolder="Results"
ft = open(ResultsFolder+"/time.txt", "a")
fr = open(ResultsFolder+"/results.txt", "a")
ReadData("TestData")
ft.close()
fr.close()