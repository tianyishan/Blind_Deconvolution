
# PRIDA_CPP

This is a cpp implementation of PRIDA

### Prerequisites

OpenCV should be installed 

CMake should be installed

### Running the Code 
    mkdir build
    cd build 
    cmake ..
    make
    ./prida <imagename>.png or pathname lambda kernel_size
For example:

    ./prida ../babies.png 0.0006 19
    
When the program finishes, it will write the result into the same folder of your input image.  

A future goal of this project is to optimaize the speed of running the algorithm. 
