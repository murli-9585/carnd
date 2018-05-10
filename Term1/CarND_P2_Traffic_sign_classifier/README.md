#**Traffic Sign Recognition** 
By Murali Madala.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on random images downloaded from internet.
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/images_output.png "Input Data"
[image2]: ./writeup_images/train_validation_test_histogram.png "Data histogram"
[image3]: ./writeup_images/data_preprocessing.png  "Data preprocessing"
[image4]: ./writeup_images/validation_test_performance.png "Validation & test performance"
[image5]: ./writeup_images/random_images.png "Random images downloaded from web."


## Rubric Points
 
---
###Writeup / README

Project code is located along with this file's directory. [Project Code]: ./Traffic_Sign_Classifier.html
Ipython file: [Code file]: ./Traffic_Sign_Classifier.ipynb

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set. Numpy was used to image preprocessing and analyse data sets.
### 1. Data set Summary:
* Input= (34799, 32, 32, 3)
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes = 43

####2. Include an exploratory visualization of the dataset.

Data is colored with all images of smae size 32X32X3. Here are 3 random sample orignal images.

![alt text][image1]

Histogram of training samples VS Validaiton and Test samples is below:
![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 
Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Data is loaded and analysed wrt to counts, sizes and required signs data for comparision.
I did pre-processing of data so it helps with performance in calcuating CNN's (Convolutions neural network) later. 

First, the images were converted to gray scale. Technique used is simple averaging of RGB color layers of original image. Gray scale imaging in this case is beneficial because:
	i) Calculations are quicker compared as image has 1 layer compared with 3 with original.
	ii) Color, position of image does not matter in classifying, so having gray scale will have image size reduced to 1/3.
 
Further pre-processing is done by normalizing the image  to have mean zero and equival variance.
Pixels are centered between -1 to 1 with a mean value of Zero. Since traffic signs are very specific data sets and not 
really complex images (containing more variables) I felt this technique would work well.

Visuvalization of original image to gray scale to normalization is showing in Data-preprocessing below.

![alt text][image3]


The difference between the original data set and the augmented data set is following after data pre-processing.
Here is the example of data for a random image. Comparision between original vs gray vs normalized is made for a random row.

5 pixels of first row for the same image (original, Gray and normalized) for comparision.
Original:
[22 22 25]
[23 23 28]
[25 25 33]
[22 21 25]
[24 23 27]
Gray
[ 23.]
[ 24.66666667]
[ 27.66666667]
[ 22.66666667]
[ 24.66666667]
Normalized
[-0.81960784]
[-0.80653595]
[-0.78300654]
[-0.82222222]
[-0.80653595]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
* Layer 1 (Convolute -> subsample(activation & pooling):
 	1.1) Convolution: The output shape should be 26X26X6
 	1.2) Activation: RELU.
 	1.3) MaxPooling: Output shape 13x13x6.

* Layer 2 (Convolute -> subsample(activation & pooling):
	2.1) Convolution: The output shape should be 7x7x16.
	2.2) Activation: RELU.
	2/3) Pooling(Max or TODO): The output shape should be 3x3x16.

* Flatten from X*Y*Z to N; 144.
 	Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
 
* Layer 3 (Fully Connected -> Activation)
	Fully connected layer 1. This should have 120 outputs.
	Activation: RELU

* Layer 4 (Fully Conected - > Activation)
	Fully connected layer 2. This should have 84 outputs.
	Activation 4. RELU.

* Layer 5 (Fully connected; Logits)
	Fully connected layer 3. This should have 43 outputs.


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
Training the model is done using these parameters.
rate = 0.00099
EPOCHS = 20 # From the results, I feel 15 should be good here. 
BATCH_SIZE = 128
Optimizer: Max Pooling (1, 2, 2, 1)
Weights used for convolution: (1, 7, 7, depth); Depth for layer 1: 6, Layer 2: 16.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy: 100%
* Validation set accuracy: 95.6%
* Test set accuracy: 92.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	The first architecture used as LeNet with no data pre-processing. Given the success rate with MINST data it was adopted.
	
* What were some problems with the initial architecture?
	Validation accuracy stayed below 90% and execution time was more due to color layer was not removed.
	
* How was the architecture adjusted and why was it adjusted? 
	Architecture was adjusted with adding a third convolution layer so depth is same as flattened layer. Patch/kernel sizes
	were adjusted to 7X7 from 5X5. Since traffic signs are mostly meant for sign representation shapes from one image to other
	are expected to be different. So a larger kernel size might work. A comparision with 5X5 is made and after testing with test
	and random-test images I am convinced that 7X7 works well with added performance. Regularization (Max Pooling) was added
	to convolution layer so the size of indivijual sublayers are reduced by a factor of 4.
	
* Which parameters were tuned? How were they adjusted and why?
	Following parameters were tuned:
	a) Epochs were tuned to 20 as the model was training until 15 epoches and reached a stabilized point. 5 More were added to see if
	there is too much under/over fitting so futher tuning decision can be made.
	b) Learning rate is kept is 0.00099 for the model to learn at slower pace initially but re-correction is less going forward. Which can be noticed 
	with validation and test performance 

![alt text][image4]


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are Eight German traffic signs that I found on the web:

![alt text][image 5]

The images are choosen with visually varying parameters. For ex:
 a) 30Kmph image: Color varience with green background
 b) Priority road with white background. This is to validate gray /normalization (to see zero errors)
 c) Yeild sign is slightly sideways to check if orientation works.
 d) Bumpy Road and Road work with smaller images inside an image/sign.
 e) Turn right with only 2 varying colors but with high pixel coverage (Blue and white)
 f) Roundabout mandatory with the sign aligned at the very top of the image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 Kmph     		    |	 30 Kmph     								| 
| Priority road    		| Priority Road 								|
| Yield 				| Yield											|
| Bumpy Road	      	| Bumpy Road					 				|
| Road work 			| Road work          							|
| Children crossing 	| Children Crossing								|
| Right turn ahead	    | Right turn ahead				 				|
| Round about mandatory	| *Traffic signals          					|


Model predicted 7 of the 8 images correctly. With a 87.5% accuracy. This is close enough with test data set used at 93%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities were


| Image			        |     Probablities	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 Kmph     		    | .999997854e-01    							| 
| Priority road    		| 9.99991775e-01								|
| Yield 				| 9.81332660e-01								|
| Bumpy Road	      	| 9.67614472e-01					 			|
| Road work 			| 9.99999881e-01          						|
| Children crossing 	| 9.99545515e-01							    |
| Right turn ahead	    | 1.00000000e+00				 				|
| Round about mandatory	| 8.65299165e-01*          					    |
Note the "round about mandatory" image, prediceted .86 for Traffic signs and 0 for round about mandatory.




