# EAI1
Project related to the first module of "Elective in AI" Course. Topic -> AI for Visual Perception


## Required slib
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



