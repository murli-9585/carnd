## Advanced Lane Finding  ReadMe


**Advanced Lane Finding Project**

Goal of the project is to find lanes for a given image/video using advanced
techniques thought in course work.
Camera calibration and distortion techniques are used with sample images
and applied on lane images. Images are modified to get a top-down view.
This project uses Sobel gradients, HSV color schems to make lane detection easy.
Lane lines are detected using sliding window approch. Once lane lines
are found radius curvature and distance from center are detected to
understand and further refine the process.  

Here are the steps used in this solution:

1) Calibrate and undistort images by using sample chess pictures.
2) Unwarp the image.
3) Apply gradients (sobel threshold x-axis, magnitude and directional thresholds)
4) Apply HSV color schems to capture and reduce noise (HLS_S and HLS_L schemes are applied)
5) Select sample area of the image to unwarp the image.
6) Apply sliding window techniques.
6.1) Sliding window centroid: The goal was to utilize this approch as it proved
effective on images. My implementation had issues with it and the right lane
is drifting away. Approch removed from final project.
6.2) Sliding window polyfit: To get the initial points of lanes a histogram is
used and base points are detected. Next sliding rectangle windows are applied
and collet all non-zero pixels in the rectangle to better polyfit the lines.
This worked well.
7) Calcuate radius curvature and distance from center based on the lines found.
8) Test on all sample images.
9) Test on videos. 
10) Repeat 3 to 9 based on testing.

[//]: # (Image References)

[image1]: ./test_images/camera_calibration.png "Finding Object and Image points"
[image2]: ./test_images/undistort.png "Applying Calibration and Distortion"
[image3]: ./test_images/undistort2.png "Undistortion on image captured for lane."
[image4]: ./test_images/image_pipeline.png "Applying image pipeline on test images."
[image5]: ./test_images/poly_lines_sample.png "Applying src points on sample image."
[image6]: ./test_images/warped.png "warped"
[image7]: ./test_images/sliding_window_polyfit.png "Sliding window polyfit"
[image8]: ./test_images/histogram.png "Histogram for finding intial line fits."
[image9]: ./test_images/polyfit.png "polyfitting lanes"
[image10]: ./test_images/draw_lanes_with_radius1.png "Draw lanes with radius on test image"
[image11]: ./test_images/draw_lanes_with_radius.png "Lanes with radius on all Images"
[video1]: ./output_videos/project_video_output.mp4 "Project input video"
[video2]: ./output_videos/challenge_video_output.mp4 "Challenge video"
[video3]: ./output_videos/harder_challenge_video_output.mp4  "Harder challenge video"

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./eMM_advanced_lane_lines_p4.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

After identifying object and image points I applied calibration and undistorted the image. Before and after can be seen below:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Applied distortion correction to one of the Lanes and below is comparision for before and after.
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used combination of color and gradient thresholds to generate a binary image. Following transformation are applied. 
Provided are methods in code. All of this are combined in method named image_pipeline(image) of the above ipynb file.
Following are the transformations which are applied.
i) Absolute sobel threshold for X_axis. Method Name: abs_sobel_thresh()
ii) Magniture threshold. Method name: mag_threshold()
iii) Directional threshold. Method name: dir_threshold()
iv) HLS with S tranform. Method Name: hls_s()
v) HLS with L transform. Method name: hls_l(). This was perticularly useful in removing shadows.

Combined image looks like this:
This gives before and after pictures. Please note: The following images also has unwarp applied which will be discussed below.
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Source and destination points were calculated using offset. Source points were choosen constants
as all the pictures/video images captures are from center camera.
Code for this:
offset = int(orig.shape[1]/4) # 1280/4 = 320.
src = np.array([[200, 700],
                  [550, 485],
                  [750, 485], 
                  [1150, 700]], np.float32)
				  
dst = np.array([(offset, orig_h),
                  (offset, 0),
                  (orig_w-offset, 0),
                  (orig_w-offset, orig_h)], np.float32)

				  
Drawing polylines with source points would look like:
![alt text][image5]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 700      | 320, 720      | 
| 550, 485      | 320, 0        |
| 750, 485      | 960, 0        |
| 1150, 700     | 960, 720      |

Verified src and dst after warp'ing the image. Below is what it would look on all sample images. This is 
what the image will be given to further apply gradients and color schemes.
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I originally tried out with finding sliding window techniques with finding centroids. However this did not work as i
expected. Moved to simple mean arthametic polynomial fit but still using the same sliding window(rectangle) technique.

Here are the steps for both:

A) Sliding window centroids
i) Take 1/4th of image to find initial starting points by dividing the pic in half vertically.
ii) Pre-determine the sliding window height /width, margin with sizes (80, 80). Margin of 50 
iii) This method used np.convolve on sum of pixels in the rectangle and determines center points for left and right
using np.argmax.
iv) Based on rectangles pixels, collect non-zero points to apply poly fit for both lanes.
v) apply polyfit using np.polyfit. y = Ax**2+b*x+c
vi) return left/right fit and lane indices to visualize

B) Used sliding window polyfit.

The steps are similar like above but modified as needed.
i) Get histogram of half of the image. Since this is binary processed image. One half will be left and other is right.
This gives us the starting left and right points.
ii) Pre-determine the sliding window height /width, margin with sizes (80, 80). Margin of 50 and minpixels in a rectangle
to 40.
iii) Slide the windows for each iteration by the window size and collect all non-zero pixel values in the rectangle.
iv) Append all good points and rectangle data for visuvaliation.
v) Get mean for all good points in rectangle for that iteration; Use these values for next iteration for adjusting
the rectangle position.
vi) Using left and right non zero pixels within all the rectangles calculate polyfit using np.polyfit.
Example code:
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
		

After running through this step image would look like:

![alt text][image7]

Histogram for finding initial points will be like below:
![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Radius of curvature of indivijaul lanes uses following constants
  
y_meters_per_pix = y_meters/720 # y_meters_per_pix = 10 meters.
 #lane width is 12 ft = 3.7 meters 
 #Add a small delta as it was not always perfect fitting.
 # x_meters_per_pix = dst[3][0] - dst[0][0]; 960 - 320 = 640. 
x_meters_per_pix = 3.7/(lane_dist_px - 20) # Distance between lanes in bird's view.
Calculation is done in calculate_curve() method of the code file presented.
        
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*y_meters_per_pix +
                 left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*y_meters_per_pix +
                 right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
				 
Following reference from Udacity section 35 of Advanced lane finding.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Visualization was done under section "#Visualize" of the code. Poly fit with previous image is done in method polyfit_with_prev()

![alt text][image9]

Finally Lanes are drawn with Radius and distance from center is calculated; Sample image and on all images is below.

![alt text][image10]
![alt text][image11]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Video is captured for project video input and it woked well.
![alt text][video1]
[link to my video result](./output_videos/project_video_output.mp4)

Challenge video:
![alt text][video2]
[link to my video result](./output_videos/challenge_video_output.mp4)

Harder challenge video (not working correctly):
![alt text][video3]
[link to my video result](./output_videos/harder_challenge_video_output.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall the project was challenging. My lack of understanding with numpy clearly made me fail not to use centroids sliding window method and 
had to spent significant time in understanding how the transformations, fitting etc., really works.

Conceputally I think I understood it well. Which I felt good about how I applied pre-processing with all gradients and color schems. Due to lack
of timing or poor planning on my end, I could not achieve where my mental goal was set with this project.

I could spend more hours/days on making polyfitting better and actually use centroid technique or other techniques online via resarch papers.
I think I need bit more understanding in finding curves which would allow me put more debug data into it and find more bugs and fix it.

Overall adding more debugging to project and better understanding of scipy would make my understandings fit the work.

