# Background
1. This project is based on the National key research and development program：Immovable cultural relics theft prevention and damage prevention based on image processing.
2. Developed with Halcon and detected with image template matching algorithm. The program is compiled into a dynamic library and deployed on the device.
3. Image Gaussian filtering and template matching are implemented using CUDA C++ to accelerate the detection speed.
# Environment
* OS: Windows10
* CPU: Intel(R) i3-6100 @ 3.70 GHz
* Halcon: 18.11(64-bit)
* OpenCV: 4.6.0
# Description to the main document
* main.cpp:               Use this project to detect the demo of the test image
* imagePreprocess.cuh:    The implementation of image preprocessing on the GPU
* resultTransformate.cuh: The implementation of post-processing the result of tensorrt inference on the GPU
* trtInferenceRunner.h:   The implementation of the TensorRT inference class
* inference.cuh:          The implementation of Infernece class
* onnxrumtime:            The implementation of ONNXRuntime-gpu
# Make template UI
![image1](https://github.com/JHC521PJJ/match_template/blob/master/UI/pic1.png)  
![image1](https://github.com/JHC521PJJ/match_template/blob/master/UI/pic2.png)
# Maintainers
[@JHC521PJJ](https://github.com/JHC521PJJ).
# License
[MIT](LICENSE) © Richard Littauer
