# CODE THAT USES THE GIVEN DATASET TO TRAIN THE CNN MODEL

import numpy
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import glob
import cv2
from sklearn.utils import shuffle
import os

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load training data for gesture 2
myFiveTrainImageFiles = glob.glob("D:/train/fiveFingerTrainDataset/*.jpg")
myFiveTrainImageFiles.sort()
myFiveTrainImages = [cv2.imread(img,0) for img in myFiveTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myFiveTrainImages)):
    myFiveTrainImages[i] = cv2.resize(myFiveTrainImages[i],(50,50))
tn1 = numpy.asarray(myFiveTrainImages)

##################################################################################################################################START

#load training data for gesture 3
myOneTrainImageFiles = glob.glob("D:/data/onefingure/*.jpg")
myOneTrainImageFiles.sort()
myOneTrainImages = [cv2.imread(img,0) for img in myOneTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myOneTrainImages)):
    myOneTrainImages[i] = cv2.resize(myOneTrainImages[i],(50,50))
tn3 = numpy.asarray(myOneTrainImages)

# load training data for gesture 4
myTwoTrainImageFiles = glob.glob("D:/data/twofingure/*.jpg")
myTwoTrainImageFiles.sort()
myTwoTrainImages = [cv2.imread(img,0) for img in myTwoTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myTwoTrainImages)):
    myTwoTrainImages[i] = cv2.resize(myTwoTrainImages[i],(50,50))
tn4 = numpy.asarray(myTwoTrainImages)


# load training data for gesture 5
myThreeTrainImageFiles = glob.glob("D:/data/threefingure/*.jpg")
myThreeTrainImageFiles.sort()
myThreeTrainImages = [cv2.imread(img,0) for img in myThreeTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myThreeTrainImages)):
    myThreeTrainImages[i] = cv2.resize(myThreeTrainImages[i],(50,50))
tn5 = numpy.asarray(myThreeTrainImages)

# load training data for gesture 6
myFourTrainImageFiles = glob.glob("D:/data/fourfingure/*.jpg")
myFourTrainImageFiles.sort()
myFourTrainImages = [cv2.imread(img,0) for img in myFourTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myFourTrainImages)):
    myFourTrainImages[i] = cv2.resize(myFourTrainImages[i],(50,50))
tn6 = numpy.asarray(myFourTrainImages)

####################################################################################################################################END

# load training data for gesture 1
myZeroTrainImageFiles = glob.glob("D:/train/zeroFingerTrainDataset/*.jpg")
myZeroTrainImageFiles.sort()
myZeroTrainImages = [cv2.imread(img,0) for img in myZeroTrainImageFiles]

for i in range(0,len(myZeroTrainImages)):
    myZeroTrainImages[i] = cv2.resize(myZeroTrainImages[i],(50,50))
tn2 = numpy.asarray(myZeroTrainImages)


finalTrainImages = []
finalTrainImages.extend(myFiveTrainImages)
finalTrainImages.extend(myZeroTrainImages)

############################################################################################start
finalTrainImages.extend(myOneTrainImages)
finalTrainImages.extend(myTwoTrainImages)
finalTrainImages.extend(myThreeTrainImages)
finalTrainImages.extend(myFourTrainImages)

############################################################################################End

# load testing data for gesture 2
myFiveTestImageFiles = glob.glob("D:/train/fiveFingerTestDataset/*.jpg")
myFiveTestImageFiles.sort()
myFiveTestImages = [cv2.imread(img,0) for img in myFiveTestImageFiles]

for i in range(0,len(myFiveTestImages)):
    myFiveTestImages[i] = cv2.resize(myFiveTestImages[i],(50,50))
ts1 = numpy.asarray(myFiveTestImages)

# load testing data for gesture 1
myZeroTestImageFiles = glob.glob("D:/train/zeroFingerTestDataset/*.jpg")
myZeroTestImageFiles .sort()
myZeroTestImages = [cv2.imread(img,0) for img in myZeroTestImageFiles]

for i in range(0,len(myZeroTestImages)):
    myZeroTestImages[i] = cv2.resize(myZeroTestImages[i],(50,50))
ts2 = numpy.asarray(myZeroTestImages)

########################################################################################START
# load testing data for gesture 3
myOneTestImageFiles = glob.glob("D:/data/onefinguretest/*.jpg")
myOneTestImageFiles.sort()
myOneTestImages = [cv2.imread(img,0) for img in myOneTestImageFiles]

for i in range(0,len(myOneTestImages)):
    myOneTestImages[i] = cv2.resize(myOneTestImages[i],(50,50))
ts3 = numpy.asarray(myOneTestImages)



# load testing data for gesture 4
myTwoTestImageFiles = glob.glob("D:/data/twofinguretest/*.jpg")
myTwoTestImageFiles.sort()
myTwoTestImages = [cv2.imread(img,0) for img in myTwoTestImageFiles]

for i in range(0,len(myTwoTestImages)):
    myTwoTestImages[i] = cv2.resize(myTwoTestImages[i],(50,50))
ts4 = numpy.asarray(myTwoTestImages)


# load testing data for gesture 5
myThreeTestImageFiles = glob.glob("D:/data/threefinguretest/*.jpg")
myThreeTestImageFiles.sort()
myThreeTestImages = [cv2.imread(img,0) for img in myThreeTestImageFiles]

for i in range(0,len(myThreeTestImages)):
    myThreeTestImages[i] = cv2.resize(myThreeTestImages[i],(50,50))
ts5 = numpy.asarray(myThreeTestImages)


# load testing data for gesture 6
myFourTestImageFiles = glob.glob("D:/data/fourfinguretest/*.jpg")
myFourTestImageFiles.sort()
myFourTestImages = [cv2.imread(img,0) for img in myFourTestImageFiles]

for i in range(0,len(myFourTestImages)):
    myFourTestImages[i] = cv2.resize(myFourTestImages[i],(50,50))
ts6 = numpy.asarray(myFourTestImages)


########################################################################################End

finalTestImages = []
finalTestImages.extend(myFiveTestImages)
finalTestImages.extend(myZeroTestImages)

###################################################################################Start
finalTestImages.extend(myOneTestImages)
finalTestImages.extend(myTwoTestImages)
finalTestImages.extend(myThreeTestImages)
finalTestImages.extend(myFourTestImages)
###################################################################################End

x_train = numpy.asarray(finalTrainImages)
x_test = numpy.asarray(finalTestImages)

# Now preparing the training and testing outputs

y_myFiveTrainImages = numpy.empty([tn1.shape[0]])
y_myZeroTrainImages = numpy.empty([tn2.shape[0]])
y_myFiveTestImages = numpy.empty([ts1.shape[0]])
y_myZeroTestImages = numpy.empty([ts2.shape[0]])

####################################################################################Start

y_myOneTrainImages = numpy.empty([tn3.shape[0]])
y_myTwoTrainImages = numpy.empty([tn4.shape[0]])
y_myThreeTrainImages = numpy.empty([tn5.shape[0]])
y_myFourTrainImages = numpy.empty([tn6.shape[0]])

y_myOneTestImages = numpy.empty([ts3.shape[0]])
y_myTwoTestImages = numpy.empty([ts4.shape[0]])
y_myThreeTestImages = numpy.empty([ts5.shape[0]])
y_myFourTestImages = numpy.empty([ts6.shape[0]])

####################################################################################End

for j in range(0,tn1.shape[0]):
    y_myFiveTrainImages[j] = 5

for j in range(0,ts1.shape[0]):
    y_myFiveTestImages[j] = 5

for j in range(0,tn2.shape[0]):
    y_myZeroTrainImages[j] = 0

for j in range(0,ts2.shape[0]):
    y_myZeroTestImages[j] = 0


#################################################################################start
for j in range(0,tn3.shape[0]):
    y_myOneTrainImages[j] = 1

for j in range(0,ts3.shape[0]):
    y_myOneTestImages[j] = 1


for j in range(0,tn4.shape[0]):
    y_myTwoTrainImages[j] = 2

for j in range(0,ts4.shape[0]):
    y_myTwoTestImages[j] = 2


for j in range(0,tn5.shape[0]):
    y_myThreeTrainImages[j] = 3

for j in range(0,ts5.shape[0]):
    y_myThreeTestImages[j] = 3


for j in range(0,tn6.shape[0]):
    y_myFourTrainImages[j] = 4

for j in range(0,ts6.shape[0]):
    y_myFourTestImages[j] = 4

#################################################################################end

y_train_temp = []
y_train_temp.extend(y_myFiveTrainImages)
y_train_temp.extend(y_myZeroTrainImages)

##########################################################################start
y_train_temp.extend(y_myOneTrainImages)
y_train_temp.extend(y_myTwoTrainImages)
y_train_temp.extend(y_myThreeTrainImages)
y_train_temp.extend(y_myFourTrainImages)

###########################################################################end

y_train = numpy.asarray(y_train_temp)

y_test_temp = []
y_test_temp.extend(y_myFiveTestImages)
y_test_temp.extend(y_myZeroTestImages)

#############################################################################start

y_test_temp.extend(y_myOneTestImages)
y_test_temp.extend(y_myTwoTestImages)
y_test_temp.extend(y_myThreeTestImages)
y_test_temp.extend(y_myFourTestImages)

#############################################################################end
y_test = numpy.asarray(y_test_temp)

print(x_train.shape)
#print(x_test.shape)

print(y_train.shape)
#print(y_test.shape)

#shuffling the data
x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

# flatten 50*50 images to a 2500 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print("num_classes")
print(num_classes)
# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=20, verbose=2)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# Save the model
model_json = model.to_json();
with open("trainedModel-5.json","w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("modelWeights-5.h5")
print("Saved model to disk")
