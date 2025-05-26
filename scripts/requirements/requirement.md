## 需要的内容
### 实现下棋任务（棋子的移动）
1. 正常的棋子移动：棋盘格点之间的移动，棋盘-->棋盘  
2. 吃子：只移动被吃的子：棋盘-->棋盘外  
我搞了一个脚本，用于移动棋子，它会订阅一个ros消息 `"kinova_grasp/target_position_in_camera"`，消息类型为`TargetPositionInCamera`

消息类型如下，详见`TargetPositionInCamera.msg`
```
string object_id # 可选
geometry_msgs/Point position_in_camera # 物体中心在相机坐标系中的位置 (x,y,z)
```

机械臂移动使用的消息类型如下，详见`kinova_grasp.py`  
```python 
    self.target_position_sub = rospy.Subscriber(
        "kinova_grasp/target_position_in_camera", # 新的话题名称
        TargetPositionInCamera,                  # 新的消息类型
        self.target_position_callback,
        queue_size=1
        )
```
然后实现棋子的移动，关于夹爪的orientation规划还没有写，后面会补充。当前的思路是：连接该棋子和他周边一圈内所有棋子的中心点，找到空隙最大的一个方向，然后去抓取。  

### 棋子移动规划 需要完成的
1. 实现一个ros节点，发布机械臂应该如何顺应象棋的规则进行移动的消息，消息类型如上面所示  
2. 包括直接移动和吃子的逻辑，需要结合前后两个棋局判断  

### 棋子识别视觉 需要完成的
1. 返回每个棋子的像素位置带标签，标签就是为了和上一个任务对接，我们需要知道控制哪个棋子  


### 我需要完成的（机械臂视觉控制）
1. 目前已完成棋盘的标定（还没有测试，我们要自己画一个棋盘）  
2. 完成了抓取的脚本（同样没有测试）

---
**我们周二/三可以再对接一下，大家都先头脑风暴一下，有问题随时在群里提，毕竟我们的时间不多了，这个周末就一定要出第一版的demo了，下周三就要结题了。**  
### 大家加油！！！