
# PRIDA_CPP

PRIDA is developed by the lab of computer vision in University of Wisconsin Madison. It stands for Provably Robust Image Deconvolution Algorithm, a image deblurring algorithm. 

This code is a C++ reliaztion of PRIDA. 

PRIDA is Provably Robust Image Deconvolution. A blind deconvolution algorithm. 

The complete implementation is recorded in [PRIDA](https://github.com/sravi-uwmadison/prida)

The paper is recorded in [arxiv](https://arxiv.org/abs/1803.08137)

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
### Result 
![babies](https://user-images.githubusercontent.com/14845016/46645630-6c9cc800-cb4b-11e8-82c7-934c3c1776fb.png)![babiesrecov](https://user-images.githubusercontent.com/14845016/46645631-6c9cc800-cb4b-11e8-84bf-61954c233003.png)


A future goal of this project is to optimaize the speed of running the algorithm. 
