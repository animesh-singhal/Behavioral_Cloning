# **Behavioral Cloning** 

## Aim

### Train a deep neural network to drive a car like you!

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3, 5x5 filter sizes and depths between 24 and 64. The neural network architecture that I've used is NVIDIA's research model with some minor changes. 

The model includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer. The images are cropped too desired dimensions using a Keras Cropping2D layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I've added some dropout layers in betwen the fully connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I had tried collecting a lot of training data as discussed below but eventually, it wasn't giving the desired results. I eventually used the training data provided by Udacity to get the job done. I even tried combining both the data sets in an attempt to increase the performance further but the model degraded further.

But this is how I collected the training data: 
I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the backward direction, multiple laps of the track and challenge track (with the same types of data sets as mentioned above for the regular track).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have an iterative approach. 

I first pre-processed my data and quickly got a very basic model working initially with a densely connected layers. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Next step was to normalise the data which resulted in a drastic reduction of both: validation and test set loss. This was achieved by adding a lambda layer. At this stage, my car got started but was having trouble driving even forward with confidence. 

To further improve the architecure I used LeNet because it is known to do a good job with images. It brought down the errors to very low values. 

I tried testing this model by running the simulator to see how well the car was driving around track one. The model atleast managed to get the car driving straight without much deviations. But it was failing to pass through the simple low curvature turns. It fell into the lake a few times too with such a raw model! 

To increase the performance further, I used 2 techinques to increase the number of samples in my data set: 
1) Image flipping: I flipped each training set image about a vertical line passing through the centre of the image. 
2) Multiple camera images: So far, I had only taken the input from the centre camera mounted on the car. But, we also had 2 more camera mounted on the car capturing the left and the right view. With minor modifications to the steering angle value (to make it consistent for the other two cameras), I added these images (with modified labels) to the training set

Now after applying this, the car gracefully completed the track passing all the curves very nicely. 

I further replaced LeNet with an advanced architecture which was designed by NVIDIA. Implementing this model gave good results too. In one temporary interation, when I removed the images obtained from the left and the right camera and just tried the NVIDIA model, the car ran pretty well on the entire track running well on all the turns. 

Now, in order to reduce the excessive memory utilization while training, I used the concept of generators to train the model using mini batches in each iteration. This obviously had a bit of detrimental effect on the performance of the car in the autonomous mode, but the technique was important from a future application to huge databases standpoint. The reason for the slight decrease in performance for simple: instead of training the whole dataset at once in an epoch, training was done on small batches which won't capture the complete picture at once. But, the model tries it best to optimise each mini batch it runs though in it's course of training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I think that the result could be even improved by collecting some more quality data. 

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes followed by deeply connected layers as shown below:

`model.add(Lambda( lambda x: x/255 -0.5, input_shape=(160,320,3) ))
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
model.add(Dense(1))`

Here is a visualization of the architecture (The images contains the model from NVIDIA's paper. My model is a close replica of this model with a few additional layers)

<img src="examples\NVIDIA_model.png" width="350">

