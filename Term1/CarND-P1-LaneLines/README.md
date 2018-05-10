#**Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds the lane lines on the road.
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteCurve.jpg "Solid white curve"
[image3]: ./test_images_output/solidWhiteRight.jpg "Solid white right"
[image4]: ./test_images_output/solidYelloeCurve.jpg "Solid Yellow curve"
[image5]: ./test_images_output/solidYelloeCurve2.jpg "Solid Yellow curve - 2"
[image6]: ./test_images_output/solidYellowLeft.jpg "Solid Yellow left"
[image7]: ./test_images_output/whiteCarLaneSwitch.jpg "Switch lane"

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
1. Convert the image to grey scale; The pixel count on the image is reduced to 2pow8 which also helps with identifying with
lane colors.
2. Apply Gaussian smoothing; Reduce noise on the image. I tried out with low_threshold of 60 and high_threshold of 180; Giving it a 1:3 ratio.
3. Apply canny edge on the smoothed image to indentify solid structures.
4. Select a quadrilateral polygon to mask the edges unwanted edges. 
5. Apply hough transform on masked polygon/edges. This involves joing edges. This is basically done by extrapolating the lines using
following (note: I could have used numpy API; more in shortcomings):
5.a)  Add slope, y-intercept and line_length for each line using y=mx+b
5.b) Given the slope, differenciate positive and negative lines.
5.c) Given the line length perform weighted average by selecting lines with larger lines lengths.
5.d) Calculate average slope, y-intercept using large lines.
5.e) Using average slope, y-intercept extrapolate both positive and nagative upto 60% of image length. So a single line
is drawn for both negative and positive lanes.
6. Overlay the line image on original imgage.

Here are the sample output images.
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]


###2. Identify potential shortcomings with your current pipeline

Potential shortcomings:
1) Lines can intersect on curvy road or when the speed is increased. 
2) Processing is done using python leading to unable to fully utilize advanced API's for applying changes on matrix(lines).
3) Smoothing could be applied at the end.



###3. Suggest possible improvements to your pipeline
Possible improvements that I could add to pipeline.
1) Apply smoothing so videos are less noisy.
2) Apply numpy functions for ex: calculating averages, Numerical application on lines (matrix) could have been easy.
