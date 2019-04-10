## Use Deep Learning to Clone Driving Behavior
##import libraries
import csv
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, MaxPooling2D, Cropping2D, Flatten, Conv2D,Dropout
from sklearn.model_selection import train_test_split
from scipy import ndimage
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

path ='/home/workspace/CarND-Behavioral-Cloning-P3/data/'
## read the csv file
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
        
## csv :  image_center image_left image_right steering throttle brake speed

correction =0.2
## Generator to read the training and validation data
def generator(samples, batch_size, shuffle=True):
    sample_len = len(samples)
    #print(sample_len)

    while 1:
        for offset in range(0,sample_len,batch_size):
            batch_sample = samples[offset: offset+ batch_size]
            images =[]
            measurements =[]
            for row in batch_sample:
              
                steering_center = float(row[3])
                steering_left =  steering_center+ correction
                steering_right =  steering_center - correction
                image_center = ndimage.imread(path+row[0].replace(" ", ""))
                #print(image_center.shape)
                #image_center = np.expand_dims(image_center, axis=0)
                #print(image_center.shape)
                image_left = ndimage.imread(path+row[1].replace(" ", ""))
                #image_left = np.expand_dims(image_left, axis=0)
                image_right = ndimage.imread(path+row[2].replace(" ", ""))
                #image_right = np.expand_dims(image_right, axis=0)
                measurements.extend([steering_center, steering_left, steering_right])
                images.extend([image_center, image_left, image_right])
               
            images = np.array(images)
            #print(images.shape)
            measurements = np.array(measurements)
            ## image augmentaton
            image_flipped = np.fliplr(images)
            #print(image_flipped.shape)
            measurement_flipped = -measurements
            images_d = np.concatenate((images, image_flipped))
            measurements_d = np.concatenate((measurements, measurement_flipped))
            
            yield sklearn.utils.shuffle(images_d, measurements_d)

            
## Training and Validation Generators
a = (train_samples[1][0])
j = ndimage.imread(path+a.replace(" ", ""))
print(j.shape)
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size, shuffle =True)
validation_generator = generator(validation_samples, batch_size=batch_size, shuffle =False)


## Defining the model 
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#print(input_shape)
model.add(Lambda(lambda x: x/127.5-1.0))

model.add(Conv2D(24,(5,5),activation ="relu")) #86 * 316
model.add(MaxPooling2D()) # 43* 158
model.add(Conv2D(36,(5,5),activation ="relu")) # 39 * 79
model.add(MaxPooling2D()) #20* 40
model.add(Conv2D(48,(5,5),activation ="relu")) # 16*20
model.add(MaxPooling2D()) #8*18
model.add(Conv2D(64,(3,3),activation ="relu")) #6*16

model.add(Conv2D(64,(3,3),activation ="relu")) #4*14
model.add(MaxPooling2D()) #2*7
model.add(Dense(32, input_dim=896))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',metrics={'output_a': 'accuracy'})

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=(len(validation_samples)/batch_size), 
            epochs=20, verbose=1)
model.save('model.h5')
print(history_object.history.keys())
