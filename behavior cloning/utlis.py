import pandas as pd
import numpy as np
import cv2
import os
import random
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam


def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    colums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log1.csv'), names=colums, skipinitialspace=True, skiprows=1,engine="python")
    data['Center'] = data['Center'].apply(getName)
    print(data.shape[0])
    return data

def balanceData(data,display=True):
    # the number of bins ploted
    nBins = 15
    samplesPerBin = 500
    hist,bins = np.histogram(data['Steering'],nBins)
    # hist is the numbers of the points that lie in each range
    # bins is the array that stores each range
    # print(bins)
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        plt.bar(center,hist,width = 0.06)
        # draw a line that starts from point (-1,samplesPerBin) to the point (1,samplesPerBin)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        removeIndexList.extend(binDataList)
    print('Removed Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print('Remaining Images: ', len(data))
    if display:
        hist, _ = np.histogram(data['Steering'],nBins)
        plt.bar(center,hist,width = 0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    return data


def loadData(path,data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append(f'{path}/photo/{indexed_data[0]}')
        steering.append(indexed_data[3])
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath,steering

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = - steering
    return img, steering

def preProcess(img):
    img = img[28:63, :, :]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
            print(steering)
        yield np.asarray(imgBatch), np.asarray(steeringBatch)


def createModel():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    model.compile(Adam(lr=0.0001),loss='mse')

    return model

