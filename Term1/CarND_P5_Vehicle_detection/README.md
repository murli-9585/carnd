##Vehicle Detection Project
# Murali Madala. Term1. Project 5.


**Vehicle Detection Project**

The goals / steps of this project are the following:
Goal: The goal of this project is to detect vehicles on road following techniques
thought in Project 5 of Term1. 
Here is breif description of steps followed:

0) Read input and check if the input for training is appropriate - Understand input.
1) Do color schems for feature extraction.
2) Perform Histogram of Oriented gradients (HOG) feature extraction on labeled
vehicles and non-vehicles.
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html#id3
3) Perform Spatial binding feature extraction.
4) Perform histogram of colors and extract features.
5) Combine all features and perform normalization of data.
6) Randomize data and extract Train and test features.
7) Perform training using Linear SVM classifier.
8) On test images perform sliding window technique with variying sizes of windows
to capture all vehicles.
9) Predict vehicle regions and eliminate false positives and noise using 
heat map and labelling.
10) Run pipe line for project and test videos.
11) Extra: Add lane detection to pipeline and perform lane and vehicle detection.

./readme_images/final_Vehicle_detection.png
./readme_images/heat_map.png


[//]: # (Image References)
[image1]: ./readme_images/input_images.png
[image2]: ./readme_images/HOG_Features.png
[image3]: ./readme_images/testing_with_feature_params.png
[image4]: ./readme_images/vehicle_detection_using_single_region.png
[image5]: ./readme_images/vehicle_detection_using_multiple_windows.png
[image6]: ./readme_images/testing_on_all_images.png
[image7]: ./readme_images/Normalization_histogram.png
[image8]: ./readme_images/final_Vehicle_detection.png
[image9]: ./readme_images/heat_map.png

[video1]: ./output_videos/output_vehicle_project_video.mp4
[video2]: ./output_videos/output_lane_vehicle_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


---
###Writeup / README
This is document is used as README.

###Histogram of Oriented Gradients (HOG)

####0. Input images.
I parsed input images (count and their shapes) and made sure I used only one way of reading images. Thanks for Udacity tips and tricks.

Overall car files: 8792
Overall of non-car files: 8968

![alt text][image1]

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the project submission, get_hog_features method extracts HOG features. The parameters are defined in "Feature_Params" class.
HOG feature params along with color histogram and bin spatial are defined as follows.
<code snippet>
        self.color_space = "YCrCb" 
        # Final image size which is used to extract features.
        self.req_size = (32, 32)
        # HOG orient parameter.
        self.orient = 9
        # HOG pix_per_cell param
        self.pix_per_cell = (8, 8)
        # HOG cell_per_block
        self.cells_per_block = (2, 2)
        # Visualize HOG features
        self.hog_vis = False
        # Ravel before returning HOG
        self.feature_vec = True  
        # For color histograms
        self.nbins = 32
        # Range of bins; This could very well be detected. For now constant.
        self.bins_range = (0, 256)
		
</code snippet>

Using following paramters above, I applied HOG to car and non-car images.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` 
and `cells_per_block=(2, 2)`
HOG feature extraction for vehicle and non-vehicle images is shown below:

![alt text][image2]

Apart from HOG features, I also extracted bin spatial and Histogram of colors feature extraction techiques so 
feature set is rich.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and collected manual tracking set 
![alt text][image3]

Seconds to train SVC is different to execution time noted above. Execution time above
is to extract features with training SVC.

I felt accuracy was consistent with YCrCb. On the last run results look like:
Training Features: 14208
Test Features: 3552
Seconds to train SVC: 7.0
Test Accuracy of SVC: 0.9882

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in "linear_svc" method with c_param of 100.0.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I tried with scale 1.5 across image Y-axis range of:
y_low_high = (375, 656)
scale = 1.5
Output looked like:
![alt text][image4]

Next, I applied same image with varying scales and regions below:
  y_low_highs = [(340, 656), (350, 656), (375, 656), (400, 656)]
    scales = [1.0, 1.5, 2.5, 3.5]
	
When passing test image through the classifier it looked like:
![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

Applied normalization on input using StandardScalar techique making features expand across. Raw features vs normalized features would
look something like this:
![alt text][image7]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Project video is processed with just vehicle detection and it detects vehicles correctly.
![alt text][video1] 

Challenge:
Applied Advanced lane finding along with vehicle detection on project video and this is how it looks.
Note: There is noise in this video due threshold was low on this one. The amount of time it takes ~ 1hr made me
to use it and not re-run a new test. However, the video for vehicle detection above uses with updated threshold.

![alt text][video2]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is one of the frames and their corresponding heatmaps:

![alt text][image9]

### Here the resulting bounding boxes are drawn onto the test image the series:
![alt text][image9]


### Steps for performance improvement
Following attempts (1. successful and 1. failure) to enhance performance were made.
1) Use multi threading to process advanced lanes and detecting vehicles. This reduced execution time from ~1.45 mins to ~1hr.
2)  Use Previous labeled boxes for searching next frame.
Searching each frame of video with sliding window is costly. It goes over each frame
with 3 scales with 3 different window sizes. Then uses heat map and labelling to condense to labaled boxes. This is costly. 
We could re-use previous frames labeled boxes and extrapolate
them by a margin of ~10 pixes (left, right and forward) and then use predict we can determine
hot space without too much calculation. When prediction is below certain % we can switch back to multi scaling. 

This could be previously implemented but I am not sure. After reading through https://dam-prod.media.mit.edu/x/files/thesis/2015/savannah-ms.pdf

https://www.polygon.com/2014/6/5/5761780/frame-rate-resolution-graphics-primer-ps4-xbox-one
I felt this is a good performance ehnacement thing to do;

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Very exciting and challenging project. I felt learned and high productive during these 2 weeks :) However, here are my thoughts:

1) I notice the 30 second video took almost 45mins to perform lane and vehicle detection. This is super poor performance but one positive side
I learned a lot and this gave a very good idea on how detection works.
2) Trying out classifier created in "Traffic signs" which does deep learning might be a good idea or other deep learning classifier.
3) Reading more published papers and their approchs could save time in following workflow and writing code might be a great idea.
4) I am not confident that my pipeline will work as intended in rain/dark areas. 

