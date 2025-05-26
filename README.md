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

---
## 关于视觉的部分
1. 相机的aruco标签识别已经写好了，运行 `aruco_detector_node.py` 即可通相机视觉标定aruco标签，并且发布视觉标签的原点坐标在 `/aruco_detector/markers ` 中。
2.  相机的坐标系标定也写好了，运行 `forward_kinematics.py` 即可通过marker发布的数据计算目标位置的cartesian coordinate。
3.  使用 `moveit.py` 即可控制机械臂移动现移动到top_view再移动到目标位置（这里需要手动设置目标点的坐标，后面会修改为直接调用forward kinematics的结果不用手动调包了。
---
- 相机坐标系相对位置  
$$
^6_{camera}T = \begin{bmatrix}
   0 & -1 & 0 & 60 \\
  1 & 0 & 0 & 0 \\
  0 & 0 & 1 & -110 \\
  0 & 0 & 0 & 1
  \end{bmatrix}
$$

- top_view 关节位置
`top_view_pos = [30.66, 346.57, 72.23, 270.08, 265.45, 345.69]`


---
## 关于抓取
1. `board_loc.py`是给棋盘定位的，返回的是中心点和朝向，暂时用不到  
2. `kinova_grasp.py`是抓取的脚本，接受`TargetPositionInCamera`消息然后实现机械臂的抓取。这个文件基于了`forward_kinematics.py`以及`move_cartesian.py`，分别完成手眼变换以及对机械臂实体的控制。  
3. 接下来考虑抓取规划：优化夹爪朝向，避免碰到周围的棋子  
### 实现下棋任务
1. 正常移动棋子：棋盘->棋盘  
2. 被吃的子：棋盘->棋盘外  
这两步让AI和视觉的同学完成  