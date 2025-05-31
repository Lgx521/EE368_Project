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
3.  使用 `moveit.py` 即可控制机械臂移动现移动到top_view再移动到目标位置（这里需要手动设置目标点的坐标，后面会修改为直接调用forward kinematics的结果不用手动调包了)。
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

---
## 抓取测试
面向`kinova_grasp.launch`与`kinova_grasp.py`的测试  
目前就是传抓取地点的消息，示例消息如下：  
```shell
rostopic pub -1 /kinova_pick_place/goal_in_camera ee368_project/PickAndPlaceGoalInCamera \
'{
  object_id_at_pick: "red_pawn_A1",
  pick_position_in_camera: {x: 0.0, y: 0.0, z: 0.42},
  target_location_id_at_place: "empty_square_D4",
  place_position_in_camera: {x: 0.12, y: 0.12, z: 0.42}
}'
```

### 棋盘角点检测
角点检测在`chessboard_detector_node.py`中，其包含了自定义的消息类型分别发布四个棋盘角点在相机坐标系中的cartesian位置。  
发布话题在`/chessboard_corners`，目前发布频率为30Hz，可调。  

### AI下棋以及移动消息发送
已经merge到main中了，实现AI下棋与移动消息发送，还未测试。  
---


## chess_ai_node.py的使用

这个节点接收两个消息，分别是棋盘的四个角点坐标和棋盘矩阵，并且发布一个消息，即需要移动的棋子的当前坐标和目标坐标。在接收到两个消息之前节点不会发布任何消息。在接收到更新后的棋盘矩阵之前它也不会发布新消息。
下面是如何发布这两个消息的示例。

### 角点坐标

```shell
rostopic pub /chessboard_corners ee368_project/ChessboardCorners "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
top_left:
  x: -0.17
  y: 0.07
  z: 0.63
top_right:
  x: -0.18
  y: -0.13
  z: 0.63
bottom_left:
  x: 0.18
  y: 0.06
  z: 0.67
bottom_right:
  x: 0.16
  y: -0.15
  z: 0.65"
```

### 棋盘矩阵
```shell
rostopic pub /chess_board_matrix std_msgs/String "data: \"[['r','n','b','a','k','a','b','n','r'], ['0','0','0','0','0','0','0','0','0'], ['0','c','0','0','0','0','0','c','0'], ['p','0','p','0','p','0','p','0','p'], ['0','0','0','0','C','0','0','0','0'], ['0','0','0','0','0','0','0','0','0'], ['P','0','P','0','P','0','P','0','P'], ['0','0','0','0','0','0','0','C','0'], ['0','0','0','0','0','0','0','0','0'], ['R','N','B','A','K','A','B','N','R']]\""
```
这个示例的含义是初始时红方的炮移动到棋盘中央。如果发布初始的矩阵，节点不会有任何反应。因为 AI 使用的是黑方（小写字母），是后手。
