import os
import csv

print("Reading rows from the data sheet...") 
samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter=0
    for line in reader:
        if (counter!=0):            #The counter is used to ignore the header columns in our data sheet
            samples.append(line)
        counter=1

"""        
with open('/home/workspace/CarND-Behavioral-Cloning-P3/turn_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('/home/workspace/CarND-Behavioral-Cloning-P3/turn_data_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('/home/workspace/CarND-Behavioral-Cloning-P3/chal_track/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('/home/workspace/CarND-Behavioral-Cloning-P3/reverse_run/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
"""        
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
from scipy import ndimage
import numpy as np
import sklearn
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction_term=0.4
            for batch_sample in batch_samples:
                for i in range(3):
                    
                    source_path = batch_sample[i]
                    filename= source_path.split('/')[-1]     #Here we are taking the centre image from the sample
                    
                    if (len(source_path.split('/'))==2):
                        current_path = '/opt/carnd_p3/data/IMG/'+ filename
                    """
                    elif (source_path.split('/')[3]=='turn_data'):
                        current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/turn_data/IMG/' + filename
                    elif (source_path.split('/')[3]=='curve_data_2'):
                        current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/turn_data_2/IMG/' + filename
                    elif (source_path.split('/')[3]=='reverse_run'):
                        current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/reverse_run/IMG/' + filename
                    elif (source_path.split('/')[3]=='chal_track'):
                        current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/chal_track/IMG/' + filename
                    """
                    
                    image = ndimage.imread(current_path)
                    images.append(image)

                    if i==0:
                        angle = float(line[3])  #4th column has steering data
                    if i==1:
                        angle = float(line[3])+correction_term
                    if i==2: 
                        angle = float(line[3])-correction_term
                    angles.append(angle) 
                    
            augmented_images = []
            augmented_angles = []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                image_flipped = np.fliplr(image)
                angle_flipped = -angle
                augmented_images.append(image_flipped)
                augmented_angles.append(angle_flipped)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
print("Preparing the generators...")
train_generator = generator(train_samples, batch_size=batch_size)               #This output will go in model.fit directly
validation_generator = generator(validation_samples, batch_size=batch_size)

print("Building your network...")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda( lambda x: x/255 -0.5, input_shape=(160,320,3) ))
model.add(Cropping2D( cropping= ((70,25), (0,0)) ))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

import math

model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')
model.save_weights('weights.h5')    