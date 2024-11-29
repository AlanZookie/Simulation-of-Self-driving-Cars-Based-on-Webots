"""Sample Webots controller for highway driving benchmark."""
import os
import cv2
import shutil
import numpy as np
from vehicle import Driver
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.image as mpimg
from io import BytesIO


def preProcess(img):
	img = img[28:63, :, :]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	img = cv2.resize(img, (200, 66))
	img = img / 255
	return img

maxSpeed = 50
minSpeed = 20
speed = 30
driver = Driver()
driver.setSteeringAngle(0.0)  # go straight
# path
imgPath = 'D:\\_1Asenior\\gra_pro\\four\\imdd.png'

# get the camera
camera = driver.getDevice('front_camera')
# uncomment those lines to enable the camera
camera.enable(50)
camera.recognitionEnable(10)
while driver.step() != -1:
	# path
	model = load_model('D:\\_1Asenior\\gra_pro\\four\\model1.h5')
	camera.getImage()
	camera.saveImage(imgPath, 100)
	img = mpimg.imread(imgPath)
	img = np.asarray(img)
	img = preProcess(img)
	img = np.array([img])
	steering = float(model.predict(img))
	driver.setSteeringAngle(steering)
	numberOfObjects = camera.getRecognitionNumberOfObjects()
	if speed > maxSpeed:
		speed_limit = minSpeed  # slow down
	else:
		speed_limit = maxSpeed
	if (camera.hasRecognition):
		my_object = camera.getRecognitionObjects()
		for i in range (numberOfObjects):
			if 'building' in my_object[i].get_model().decode():
				print(my_object[i].get_model())
				driver.setCruisingSpeed(30)
			else:
				driver.setCruisingSpeed(40)
	else:
		driver.setCruisingSpeed(40)
	speed = driver.getCurrentSpeed()
	print('{} {}'.format(steering, driver.getCurrentSpeed()))
# adjust the speed according to the value returned by the front distance sensor
# frontDistance = sensors['front'].getValue()
# frontRange = sensors['front'].getMaxValue()
# speed = maxSpeed * frontDistance / frontRange
# driver.setCruisingSpeed(speed)
# brake if we need to reduce the speed
# speedDiff = driver.getCurrentSpeed() - speed
# if speedDiff > 0:
# driver.setBrakeIntensity(min(speedDiff / speed, 1))
# else:
# driver.setBrakeIntensity(0)
