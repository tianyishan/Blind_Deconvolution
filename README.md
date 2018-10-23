
# Blind_Deconvolution

PRIDA is developed by the lab of computer vision in University of Wisconsin Madison. It stands for Provably Robust Image Deconvolution Algorithm, a image deblurring algorithm. 

PRIDA is similar in spirit to the MD algorithm in Convex Optimization. The main difference between the standard MD algorithm and PRIDA is that the step size is chosen independently for each coordinate. 

This code is a C++ realization of PRIDA. The matlab implementation is recorded in [PRIDA](https://github.com/sravi-uwmadison/prida) The paper is recorded in [arxiv](https://arxiv.org/abs/1803.08137).

## Prerequisites

CMake should be installed 

OpenCV should be installed -> [Install-OpenCV](https://github.com/jayrambhia/Install-OpenCV)

## Compile 
```script
 git clone https://github.com/tianyishan/PRIDA_CPP.git
 cd PRIDA_CPP
 mkdir build
 cd build 
 cmake ..
 make
```
## Running the Code
The format of command-line argument is:

    ./prida <imagename>.png or pathname lambda kernel_size

To run one of the demo pictures: 

    ./prida ../babies.png 0.0006 19

![screen shot 2018-10-08 at 23 22 03](https://user-images.githubusercontent.com/14845016/46646664-0450e500-cb51-11e8-88f5-e08545ef122b.png)

    
When the program finishes, it will write the result into the same folder of your input image.  
## Results  
<img width="225" height="225" alt="blur" src="https://user-images.githubusercontent.com/14845016/46645893-b33ef200-cb4c-11e8-8e3d-ab759df584b7.png"> <img width="225" height="225" alt="clear" src="https://user-images.githubusercontent.com/14845016/46645894-b33ef200-cb4c-11e8-875d-f1022e7306d6.png">

<img width="250" height="325" alt="blur" src="https://user-images.githubusercontent.com/14845016/46645944-f39e7000-cb4c-11e8-820c-1be5f2d5409a.png"> <img width="250" height="325" alt="clear" src="https://user-images.githubusercontent.com/14845016/46645945-f5683380-cb4c-11e8-8136-80524e4c5a4a.png">


### Next Steps

The future goals of this project are 
- To optimize the speed of running the algorithm. 
- To adapt it with modern deep learning methods.
- To work with a wider variety of blur kernels and images.
- To create a GUI.
