import csv
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

path ='/home/workspace/CarND-Behavioral-Cloning-P3/data/'
## read the csv file
lines = []

with open('./data/driving_log.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
        
## csv :  image_center image_left image_right steering throttle brake speed

correction =0.2

def generator(samples, batch_size):
    sample_len = len(samples)
    while 1:
        for offset in range(0,sample_len,batch_size):
            batch_sample = samples[offset: offset+ batch_size]
            for row in train_samples[1:]:
                images =[]
                measurements =[]
                steering_center = float(row[3])
                steering_left =  steering_center+ correction
                steering_right =  steering_center - correction
                image_center = mpimg.imread(path+row[0].replace(" ", ""))
                image_left = mpimg.imread(path+row[1].replace(" ", ""))
                image_right = mpimg.imread(path+row[2].replace(" ", ""))
                images.extend([steering_center, steering_left, steering_right])
                measurements.extend([image_center, image_left, image_right])
                images = np.array(images)
                measurements = np.array(measurements)
            yield sklearn.utils.shuffle(images, measurements)

            
## Training and Validation Generators
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
out = train_generator
print(out)