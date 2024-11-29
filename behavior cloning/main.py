# #print some infomation for remind
from utlis import *
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Setting Up')
# #ignore some warnings from tensorflow
# ##Step 1:ReadData
path = "D:\\_1Asenior\\gra_pro\\four"
data = importDataInfo(path)
# ##Step 2:Visualization the balanced data, remove the redundent data
data = balanceData(data, display=False)
# ##Step 3:Create the numpy data that contain the images and its steering info
imagesPath, steerings = loadData(path, data)
# ##Step 4:Separate the data into training and validation data using sklearn
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))
# ##Step 5:Augment the images to create more pictures for training
# ##Step 6:Pre-process adding Flip, pan, zoom and so on
# ##Step 7:generate more images as training samples
# ##Step 8:create model using keras
model = createModel()
model.summary()
# ##Step 9:train the model
history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0),
                    validation_steps=200)
# Step 10: Saving & Plotting
model.save('model1.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
