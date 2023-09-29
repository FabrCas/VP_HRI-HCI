
# Interlcutor’s attention-engagement estimation for HRI
Project related to the first module of "Elective in AI" Course. Topic: AI for Visual Perception.
This work proposes:

 1. Novel approach for attention estimation combining gaze-related attention and engagement (psychological).
 2. New Custom *ResNet* version for spatiotemporal dynamics, including an experimental learning phase.
 3. Innovative data processing to improve operations on video sources, balancing, enhancing, and optimizing data storage.
 4.  Introduced a combination of state-of-the-art models and computer vision techniques to infer gaze direction.
 5. A new baseline for attention estimation using a mixed solution.

For a full understanding of the work, is suggested the reading of the [report](https://github.com/FabrCas/VP_HRI-HCI/blob/main/report.pdf)

## Install requirements
Use package installer for Python to install the dependencies.

    pip install -r requirements.txt

## Required custom & Caffe ResNet model 
These models are downloadable at following [link](https://drive.google.com/drive/folders/1fdQdsUuNcvXUasF8iqcze6N-aJUYdmku?usp=sharing). Place them in the model folder

## Required dlib
Use these commands to install from source code on Ubuntu.
First check to already satisfy these dependecies:

    sudo apt-get install build-essential cmake pkg-config
    sudo apt-get install libx11-dev libatlas-base-dev
    sudo apt-get install libgtk-3-dev libboost-python-dev

Then Compile the C++ files using cmake

    wget http://dlib.net/files/dlib-19.6.tar.bz2
    tar xvf dlib-19.6.tar.bz2
    cd dlib-19.6/
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
    sudo make install
    sudo ldconfig
    cd ..
Include dlib as a library:

    pkg-config --libs --cflags dlib-1

Optionally activate the virtual envirnment (i.e. .source ./env/bin/activate)
Compile and install the dlib module

    cd dlib-19.6
    python setup.py install

The dlib predictor can be downloade at the following link: https://www.dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Place it in the model folder.
## Application Launch
To execute the software, simply use the main file

    python main.py

Default execution uses GPU (Cuda) whether available. to change this use `--useGPU False` parameter in main call.
It's possible to switch execution mode to test mode, using `--mode test` argument, which uses the test module to compute metrics.


