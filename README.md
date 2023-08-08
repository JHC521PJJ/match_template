# Background
1. This project is based on the National key research and development program：Immovable cultural relics theft prevention and damage prevention based on image processing.
2. Developed with Halcon and detected with image template matching algorithm. The program is compiled into a dynamic library and deployed on the device.
3. Image Gaussian filtering and template matching are implemented using CUDA C++ to accelerate the detection speed.
# Equipment environment
* OS: Windows10
* CPU: Intel(R) i3-6100 @ 3.70 GHz
* Halcon: 18.11(64-bit)
* OpenCV: 4.6.0
# Description to the main document
* gaussianBlur.h: The CPU implements Gaussian filtering
* gaussianBlurCuda.cuh: The GPU implements Gaussian filtering
* matchTemplate.cpp: The implementation of template matching on the CPU by Halcon
* matchTemplate.cu:  The implementation of template matching on the GPU
# Make template UI
![image1](https://github.com/JHC521PJJ/match_template/blob/master/UI/pic1.png)  
![image1](https://github.com/JHC521PJJ/match_template/blob/master/UI/pic2.png)
# Maintainers
[@JHC521PJJ](https://github.com/JHC521PJJ).
# License
[MIT](LICENSE) © Richard Littauer
