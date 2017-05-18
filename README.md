# Learning To Drive Using Behaviorial Cloning

![Autonomous Driving](writeup-resources/go-autonomous.png)

Overview
---
This repository contains code and other resources for Behaviorial Cloning project for [Udacity Self-driving Car Nano Degree](https://in.udacity.com/course/self-driving-car-engineer-nanodegree--nd013/).

The aim of the project is to train a Deep Convolutional Neural Network(CNN) model to drive a car in a simulator by learning from data gathered by a human driver driving in the same simulator. While a simulator could produce lots of data, like steering angle, speed, throttle etc, this project focused on learning to predict the steering angle by looking at images captured from front facing cameras mounted on the car. 

The project had the following stages:
* Driving data was collected from a [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip) provided by Udaicty.
* A convolution neural network was built in Keras that predicts steering angles from images
* The model was trained on a subset of the images and steering angles from the collected data then tested on another subset of the data.
* The model was tested by letting it drive several laps in the simulator.
* Results were summarized in this README file.

### Dependencies
The following dependencies need to be installed to reproduce the results dicussed here.

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

* Udacity Simulator
    * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
    * [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
    * [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)


## Details About Files In This Directory

### `model.h5`
Saved model file.

### `drive.py`

This file drives the car in autonomous mode. Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. 

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the simulator.

Screenshots of autonomous driving can be saved to a folder. To do so pass a fourth argument in the command above.

```sh
python drive.py model.h5 <snapshots>
```

The fourth argument `snapshots` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

### `video.py`
Then a video can be generated from the screenshots using the following command:


```sh
python video.py snapshots
```

Create a video based on images found in the `snapshots` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `snapshots.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py snapshots --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

### `model.py`
This file generates a Keras model from training data. The syntax for training is:

```sh
python model.py [<training_dat_dir>] [<model>] [<epochs>] [<train_batch_size>] [<test_batch_size>] [<model_name_suffix>]
```
- **training_dat_dir** is the directory where training images and driving logs are present. This directory should have a directory named `IMG` containing all training images and a file name 'driving_logs.csv' containing the driving logs generated by the simulator. If the parameter is omitted then the path `./training-data` will be used by default.

- **model** The CNN model architecture that will be trained. This can be `nvidia` which is the default or `lenet5` and is case insensitive.

- **epochs** The number of epochs to train for. Depends on the size of training data. Default is 5. Both models starts to overfit after 5 epochs.

- **train_batch_size** Size of one training batch. Default is 32.

- **test_batch_size** Batch size for testing. Default is 1 since the simulator asks for prediction on a single frame at a time.

- **model_name_suffix** By default the generated model is saved as simply model.h5. If a suffix is passed the file will be named `model-model_name_suffix.h5`


### `keras-vis.py`
This file generates activation maps from a model.h5 file. To generate the activation maps run the following command:

```sh
python keras-vis.py [<model>] [<img_dir>] [<layer>]
```
- **model** is the path to a model.h5 file.
- **img_dir** is the path to the directory containing images for which activation maps have to be generated.
- **layer** is an integer specifying the layer index for which activations are to be generated.

This will process each image and extract all activation maps for the image by runninng the model on the image and saving the output of the specified layer. The individual activation maps of an image are then combined into a single activation map and saved in `img_dir/responses/<model>/`. 

### `video.mp4`
A demostration video of autonomous driving.

# Model Design
This section describes how the model was built.

## Data Collection

### Simulator settings
The simulator was run at the lowest possible graphics settings. This was done to keep the CPU load low but discussions in the forums suggested that it was a good strategy to train on low resolution images.

### Data collection strategy
To collect human driving data, the simulator was run in training mode. The car was driven around the track for about 6 laps and driving data recorded. Then the direction was reversed and and another 3 laps were recorded. In addition a few short recordings were made of tricky parts of the track. The data recorded in the reverse direcion ensured that the model did not simply memorize the conditions of the forward lap and generalized well to test-time conditions. Some sharps turn were tricky and initial models would swerve wildly when negotiating them. The additional recordings helped the model stay in the middle of the road.

The simulator recorded screenshots taken from the perspective of 3 cameras mounted a the fron of the car at the left, center and a right of the hood. Alongwith these images the simulator also recorded the driving parameters at the instant an image was captured. These included steering angle, throttle and speed and were written to **driving_log.csv**.

![left camera](writeup-resources/left.jpg) 
![center camera](writeup-resources/center.jpg) 
![right camera](writeup-resources/right.jpg) 

### Loading the data
There were about 80,000 images whose paths were recorded in **driving_log.csv**. So actual images were not loaded all at once, instead they were loaded a few samples at a time using a Python generator described later. However, all the driving logs were loaded and analyzed, but only a small number of logs retained, as described below.  

In [model.py](https://github.com/farhanhubble/CarND-Behavioral-Cloning-P3/blob/08ab6742c4b76a96857c5704f97038ece75f88aa/model.py#L48), the function `load_logs()` loads the driving logs. We passed `all_camera=True` to this function so it loaded the image paths of all three cameras and steering angles.

### Preprocessing
Models trained on the raw data showed a propensity to drive straight ahead. This was becasue the training data had a hugely disproportionate instances of driving straight. The distribution of steering anlgles in the raw data looked like this:

![raw data histogram](writeup-resources/raw-data-distribution.png) 

To remove this bias the logs were equalized by calling [`drop_zero_steering()`](https://github.com/farhanhubble/CarND-Behavioral-Cloning-P3/blob/08ab6742c4b76a96857c5704f97038ece75f88aa/model.py#L73). Driving logs for 75% percent of steering angles in the range [-0.05,0.05] were dropped. The percentage and range values are configurable via training options in the [code](https://github.com/farhanhubble/CarND-Behavioral-Cloning-P3/blob/08ab6742c4b76a96857c5704f97038ece75f88aa/model.py#L270).

In the equalized logs, steering angles were corrected for images taken from left and right cameras. The simulator records images from three cameras as discussed above. The steering angles for left camera were increased reinforcing the need for a harder right turn, while the angles for right camera were reduced by a similar amount. A correction factor of 0.2 as suggested in the lecture videos, was passed to [`correct_steering_angle()`](https://github.com/farhanhubble/CarND-Behavioral-Cloning-P3/blob/08ab6742c4b76a96857c5704f97038ece75f88aa/model.py#L18) The distribution of driving anlges after the preprocessing looked much less skewed.

![processed data histogram](writeup-resources/data-distribution.png) 

