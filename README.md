# Self-Driving Car Engineer Nanodegree

This repository holds **code** for all the projects required by Udacity to complete Self-Driving car Engineer
Nanodegree program. **[Udacity Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)**.

## Coursework Overview

#### [Project 1 - Detecting Lane Lines ](CarND-P1-LaneLines)
 - **Overview:** Detect lane lines and using extrapolation technique extend lane lines.
 Detect lane lines on a video stream. Used OpencV image analysis techniques to identify lines, 
 including Hough Transforms, extrapolation  and Canny edge detection technique. 
 Skills: Computer Vision, Python.
 
#### [Project 2 - Traffic Sign Classification](CarND_P2_Traffic_sign_classifier)
 - **Overview:** Implemented and trained a deep neural network to classify traffic signs. Used TensorFlow. 
 Experimented with well-known network architectures for correctness. Performed image pre-processing and validation to guard against overfitting. 
 Skills: Deep Learning, TensorFlow, Computer Vision, Python
 
#### [Project 3 - Behavioral Cloning](CarND_P3_Behavioral_cloning)
 - **overview:** Built and trained a convolutional neural network for end-to-end driving in a simulator.
  Used Optimization techniques - regularization and dropout to generalize the network for driving on multiple tracks.
  Skills: Python, Deep Learning, Keras, Convolutional Neural Networks (CNNS).

#### [Project 4 - Advanced Lane Finding](CarND_P4_Advance_lane_finding)
 - **overview:** Implement advanced lane-finding algorithm using distortion correction, 
 image rectification, color transforms, and gradient thresholding. Find lane curvature and vehicle displacement.
 Skills: Python, Computer Vision, OpenCV
 
#### [Project 5 - Vehicle Detection and Tracking](CarND_P5_Vehicle_detection)
 - **overview:** Detect other vehicles on raod and track pipeline. Used OpenCV , histogram of oriented gradients (HOG), and support vector machines (SVM). Implemented the same pipeline using a deep network to perform detection. Evaluvated the model on video data from a automotive camera taken during highway driving.
 Skills: Python, Computer Vision, Deep Learning, OpenCV
 
 #### [Project 6 - Extended Kalman Filter](CarND_P6_Extended_Kalman_Filter)
 - **overview:** Implement the extended Kalman filter. lidar and radar measurements (Simulated) are used to detect a bicycle that travels around the vehicle. Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.
 Skills: C++, Kalman Filter.

 #### [Project 7 - Unscented Kalman Filter](CarND_P7_Unscented_kalman_Filter)
 - **overview:**  Utilize an Unscented Kalman Filter to estimate the state of a moving object. This is done
 when a noisy lidar and/or radar measurements are given. 
 Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.
 Skills: C++, Kalman Filter.
 
  #### [Project 8 - Kidnapped Vehicle](CarND_P8_Kidnapped_Vehicle)
 - **overview:** Implement particle filter. Given map location and a noisy GPS estimate of initial location find 
 kidnapped vehicle. Using a 2 dimensional particle filter with given map and initial localization information (GPS info)
 vehicle is detected.
 Skills: C++, Particle Filter.

 #### [Project 9 - PID Control](CarND_P9_PID_Controller)
 - **overview:** Implement a PID controller for keeping the car on track by appropriately adjusting the steering angle.
 Skills: C++, PID Controller
 
#### [Project 10 - Model Predictive Control - MPC](CarND_P10_MPC)
- **overview:** Implement an MPC controller for keeping the car on track by appropriately adjusting the steering angle. Differently from previously implemented PID controller, MPC controller has the ability to anticipate future events and can take control actions accordingly. Indeed, future time steps are taking into account while optimizing current time slot.
Skills: C++, MPC Controller

#### [Project 11 - Path Planning](CarND_P11_Path_planning)
- **overview:** The goal in this project is to build a path planner that is able to create smooth, safe trajectories for the car to follow. The highway track has other vehicles, all going different speeds, but approximately obeying the 50 MPH speed limit. The car transmits its location, along with its sensor fusion data, which estimates the location of all the vehicles on the same side of the road.
Skills: C++, Path Planning

#### [Project 12 - Road Segmentation](CarND_P12_Semantic_Segmentation)
- **overview:** Implement the road segmentation using a fully-convolutional network.
Skills: Python, TensorFlow, Semantic Segmentation

#### [Final project - System Integration](CarND_Final_Project_System_Integration)
- **overview:**  Self driving car in parking lot  with traffic lights and highway driving with traffic lights in simulator.
Used ROS (robot operating system) to control vehicle - PID controller and start/stop using waypoints relative to traffic light. Created a CNN to detect traffic lights and make vehicle start/stop based on Red/Yellow/Green lights. Same code is used
on Udacity self-driving car (carla) to test the code in parking lot with traffic lights.
Skills: Python, ROS, CNN 
<p align="left">
  <img src="https://cdn-images-1.medium.com/max/800/1*dRJ1tz6N3MqO1iCFzlhxZg.jpeg" width="80">
</p>