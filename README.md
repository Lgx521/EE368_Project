# Chinese Chess HCI Robot
The final course project of `EE368`.  

## Scheme
1. Eye-in-hand carema assign
2. Visual+Depth Data
3. In existance chess playing AI  
4. ArUco sign around the board  

## Problems
### Game AI: How to play
### Visual: Lively recogonizing the board 
### Carema tf: Where to move  
**Pipelines:**  
1. HOME position: determined height, parallel image plane and the plate, avoid monocular position indeterministic  
2. From center pixel to determine the chess piece's position in base frame $Z\cdot P_{uv}=K\cdot _w^cT\cdot P_w$  

**Technique:**  
1. Use ArUco sign to locate the chess board  
2. Use `pyrealsense2` module in python, distribute raw image to ros nodes for the CNN visual network
3. Use `OpenCV` module to do distortion removal and ArUco sign recogonition  
剩下的就是标定位置什么的，这个需要装好相机之后现场测量
[这里是CSDN的一个博客，如何使用pyrealsense2获得相机内参等等信息](https://blog.csdn.net/Dontla/article/details/102644909)

### Control: How to move
