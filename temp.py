from keras.models import load_model
import csv
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, MaxPooling2D, Cropping2D, Flatten, Conv2D
from sklearn.model_selection import train_test_split
from scipy import ndimage
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
#keras.backend.set_image_data_format('channels_first')
path ='/home/workspace/CarND-Behavioral-Cloning-P3/data/'
## read the csv file
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]

image = mpimg.imread(path+lines[4][0])

model = load_model('model.h5')
image_array = np.asarray(image)
steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

print(steering_angle)
