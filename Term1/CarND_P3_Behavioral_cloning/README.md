#**Behavioral Cloning** 


**Behavioral Cloning Project**
By Murali Madala.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./result_images/img_1.png "Model Visualization"
[image2]: ./result_images/img_2.png "Sample Image"
[image3]: ./result_images/img_3.png "After Random Transform"
[image4]: ./result_images/img_4.png "After applying shadow"
[image5]: ./result_images/img_5.png "After applying brightness"
[image6]: ./result_images/img_6.png "Horizantal flip"
[image7]: ./result_images/img_7.png "Cropping image"
[image8]: ./result_images/img_8.png "Resized Image"
[image9]: ./result_images/img_9.png "All angles comparision"
[image10]: ./result_images/img_10.png "Final agumented-preprocessed image"
[image11]: ./result_images/img_11.png "Input steering angles"
[image12]: ./result_images/img_12.png "Training and validation loss"
[image13]: ./result_images/img_13.png "Final steering historgram"
[image14]: ./result_images/img_14.png "Loss per batch"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* murali_3_behavioral_cloning.ipynb : Contains code to evaluavate input, 
preprocess and data agumentation and model implementation.
* drive.py for driving the car in autonomous mode; Mostly unchanged from 
origial code given by udacity. Updated Crop/resize section of code.
* model.h5 containing a trained convolution neural network.
* writeup_report.md or writeup_report.pdf summarizing the results and notes.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be 
driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The murali_3_behavioral_cloning.ipynb and murali_3_behavioral_cloning.py files
 contains the code for training and saving the convolution neural network. The 
 file shows the pipeline I used for training and validating the model, and it
  contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Model architecuture is slightly modifed from comma.ai model. Here is breif
description of the model.
1) A normalization layer on the top of the network to normalize the input images.
2) Convolutional layer with 16 feature maps of size of 8×8 with elu (Exponential 
   Linear) activation function.
3) Convolutional layer, 32 feature maps of size 5×5 with elu activation function.
4) Maxpooling layer with 2X2 strides and valid padding.
5) Convolutional layer with 64 feature maps of size 5×5 with elu activation function.
6) Maxpooling layer with 2X2 strides and valid padding.
7) Flatten layer.
8) Dropout set to 20%.
9) ELU non-linearity layer
10) Fully connected layer with 512 units and a elu activation function.
11) Dropout set to 20%.
12) ELU non-linearity layer
13) Fully connected layer with 256 units and a elu activation function.
14) Dropout set to 50%.
15) ELU non-linearity layer
16) Fully connected output layer with 1 unit.


####2. Attempts to reduce overfitting in the model

The model contains 2 maxpool layers between convolutions and dropout layers in
fully connected layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the
model was not overfitting. The model was tested by running it through the
simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Following training data was used.
1) Udacity data samples for track 1.
2) Collected training data for 2 additional laps and 1 reverse laps of Track1.
3) Some data for Track 2. More than half track. 
For lower angles, left and right camera angles were used and also image
 transformations were applied.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Overall strategy for driving a model architecture was to drive car using
simulator on both Track 1  and Track 2. 
Took Comma.ai model architecture to start with and apply modifications
as needed. Comma.ai was choosen as its proven and the fewer number of
conv-nets compared with other models compared gave the confidence that 
given the right/correct amount of training, it could be used successfully.
With right data pre-processing and agumentation this could run on CPU
easily.

####1.a Steps
Following steps were followed during development:

1) Analyze the input data. Data given from Udacity, 1-2 self trained on Track 1.
Since this involves 2 variables (a. Images b. Angle corrosponding to an Image)
its easy to understand.

2) Have a basic network (1 fully connected network) and see if car could drive
and understand any blocking issues with either Computer, application
software etc., That worked well.

3) Apply comma.ai model directly and undertand how the angles are applied wrt
 to images fed to it.

4) Start creating data-preprocessing and basic agumentation and apply. This 
was done with multiple steps a) Increase/reduce brightness
b) Add shadows c) Flip horizantally d) Crop image e) Reduce the size to 64X64.
Run time reduced from 1000 seconds per epoch to 200 seconds
with better results.

5) Apply data agumentation. For lower angled input (which is noted at
80% of input images) give 30% right and left camera.

6) Update the model with MAXPOOL2d and add another fully connected layer
to reduce overfitting (Was done in 2 steps). A hugeimporvement in
validation/train loss reduction was noticed after applying MAXPOOL2D.
Go MAXPOOL!

7) Re-do data-modeling and pre-processing between baches to normalize
 the data. Idea here is to get a bell-curve by the end of training.

8) Testing and apply enhancements.

9) Once satisfied, drive autonomously on both tracks.
Track 1 was trained well upto 25MPH but Track 2 the model was unable
to  drive successfuly some distance on less speed (9-10 MPH)

####2. Final Model Architecture

The final model architecture in model.py consisted of a convolution neural
network with 3 conv2D and 3 fully connected layers.
Details about params and sizes is visualized below:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using
center lane driving. Additional data is also captured by driving reverse laps.
![alt text][image2] 

Image pre-processing was done with
a) Transform image
![alt text][image3]
	
b) Apply shadow to 30% of input
![alt text][image4]
	 
c) Apply random brightness to 30% of input.
![alt text][image5] 
	
d) Flip horizantally 50% of input
![alt text][image6] 

e) Crop all images by removing 25 pixels from bottom
![alt text][image7] 

f) Re-size all images so processing can be quick
![alt text][image8] 

To augment the data sat especially low angled images,
I also flipped images and angles thinking that this would
create more angled driving.
![alt text][image9] 

Images were cropped and resized to get final image size of 64X64.
![alt text][image10]

Overall, I had 16050 training images.
Allocated training and validation samples by randomly shuffling the data and
 have 10% of the data into a validation set.
Training set: 14445
Validation set: 1605 

Initial steering angles for input looked like:
![alt text][image11] 

I used this training data for training the model. The validation set helped 
determine if the model was over or under fitting.
The ideal number of epochs used were 10. After training the model final
training loss and validation loss can be noted here.
![alt text][image12] 

I collected steering samples during training using during data generation
and final layout of steering angle was achieved with image
 pre-processing and agumentation. 
![alt text][image13] 
	
Overall loss per batch is also calculated during batch-end call back.
![alt text][image14] 

### Testing
Testing is done by running the model when the simulator is in autonomous
mode for both tracks.
Command: python drive.py model.h5 Track1.
Track 1 result: https://youtu.be/2vhMaozvGAE
Track 2 result: https://youtu.be/JI0QBOm47Zs
