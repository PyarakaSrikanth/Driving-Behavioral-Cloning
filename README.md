# Learning To Drive Using Behaviorial Cloning

![Autonomous Driving](writeup-resources/go-autonomous.png)

Overview
---
This repository contains code and other resources for Behaviorial Cloning project for [Udacity Self-driving Car Nano Degree](https://in.udacity.com/course/self-driving-car-engineer-nanodegree--nd013/).

The aim of the project is to train a Deep Convolutional Neural Network(CNN) model to drive a car in a simulator by learning from data gathered by a human driver driving in the same simulator.

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
