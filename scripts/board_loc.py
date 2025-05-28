#!/usr/bin/env python3

'''
用于定位棋盘，确定棋盘的位置。
棋盘四角由ArUco标签定位，id分别为0-3
'''

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from ee368_project.msg import ArucoMarker, ArucoMarkerArray # 确保这与您的包名匹配
import tf.transformations as tf_trans

class ChessboardLocator:
    def __init__(self):
        rospy.init_node('chessboard_locator_node', anonymous=False)

        # --- 参数 ---
        # 定义棋盘四角的标记ID
        # ID 0: 左上角
        # ID 1: 右上角
        # ID 2: 右下角
        # ID 3: 左下角
        # 这个约定对于一致地计算方向非常重要。
        self.corner_ids = rospy.get_param("~corner_ids", [0, 1, 2, 3])
        if len(self.corner_ids) != 4:
            rospy.logerr("参数 'corner_ids' 必须是一个包含4个整数的列表。将使用默认值 [0,1,2,3]。")
            self.corner_ids = [0, 1, 2, 3]

        self.expected_ids_set = set(self.corner_ids)
        self.marker_poses = {} # 用于存储检测到的目标标记姿态的字典 {id: Pose}
        self.camera_frame_id = "camera_color_optical_frame" # 默认值，将从消息中更新

        # --- 订阅者 ---
        rospy.Subscriber("/aruco_detector/markers", ArucoMarkerArray, self.aruco_markers_callback)

        # --- 发布者 ---
        self.chessboard_pose_pub = rospy.Publisher("/chessboard_pose", PoseStamped, queue_size=1)

        rospy.loginfo("棋盘定位节点已初始化。正在等待 ArUco 标记...")
        rospy.loginfo(f"期望的角点标记 ID: {self.corner_ids}")


    def aruco_markers_callback(self, aruco_marker_array_msg):
        self.camera_frame_id = aruco_marker_array_msg.header.frame_id
        detected_target_markers = {} # 用于存放当前帧检测到的角点标记

        for marker in aruco_marker_array_msg.markers:
            if marker.id in self.expected_ids_set:
                detected_target_markers[marker.id] = marker.pose
                # 用于调试单个标记位置
                # rospy.loginfo(f"检测到标记 ID {marker.id}位于 X:{marker.pose.position.x:.3f} Y:{marker.pose.position.y:.3f} Z:{marker.pose.position.z:.3f}")

        if len(detected_target_markers) == 4:
            # 所有四个角点标记都已检测到，开始计算棋盘姿态
            # 根据 self.corner_ids 中定义的顺序获取标记位置
            p0 = detected_target_markers[self.corner_ids[0]].position
            p1 = detected_target_markers[self.corner_ids[1]].position
            p2 = detected_target_markers[self.corner_ids[2]].position # 注意：p2点本身不直接用于X,Y轴定义，但用于确保检测到所有点
            p3 = detected_target_markers[self.corner_ids[3]].position

            points = np.array([
                [p0.x, p0.y, p0.z], # 对应 self.corner_ids[0]
                [p1.x, p1.y, p1.z], # 对应 self.corner_ids[1]
                [p2.x, p2.y, p2.z], # 对应 self.corner_ids[2]
                [p3.x, p3.y, p3.z]  # 对应 self.corner_ids[3]
            ])

            # --- 计算中心点 ---
            # 棋盘中心是四个角点坐标的平均值
            center = np.mean(points, axis=0)
            chessboard_center = Point(x=center[0], y=center[1], z=center[2])

            # --- 计算方向 ---
            # 我们将棋盘的X轴定义为从标记0指向标记1的方向 (或标记3指向标记2)
            # Y轴定义为从标记0指向标记3的方向 (或标记1指向标记2)
            # Z轴将垂直于棋盘平面。

            # 使用特定顺序: 0-左上, 1-右上, 3-左下
            # 从标记0到标记1的向量 (定义棋盘的局部 +X 方向)
            # 使用 self.corner_ids 中定义的索引来获取正确的点
            idx0 = self.corner_ids.index(self.corner_ids[0]) # 确保使用 self.corner_ids[0] 在 points 数组中的实际索引
            idx1 = self.corner_ids.index(self.corner_ids[1])
            idx3 = self.corner_ids.index(self.corner_ids[3])

            vec_x_raw = points[idx1] - points[idx0] # P_id1 - P_id0
            vec_x = vec_x_raw / np.linalg.norm(vec_x_raw)

            # 从标记0到标记3的向量 (定义棋盘的局部 +Y 方向的候选)
            vec_y_candidate_raw = points[idx3] - points[idx0] # P_id3 - P_id0
            vec_y_candidate = vec_y_candidate_raw / np.linalg.norm(vec_y_candidate_raw)
            
            # Z轴垂直于由 vec_x 和 vec_y_candidate 构成的平面
            # 叉乘顺序决定Z轴方向，通常我们希望Z轴"朝外"
            vec_z = np.cross(vec_x, vec_y_candidate)
            vec_z = vec_z / np.linalg.norm(vec_z)

            # 重新计算Y轴，使其与X和Z正交 (确保右手坐标系)
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            
            # 从这些基向量创建旋转矩阵
            # 旋转矩阵的列是新坐标系基向量在原始坐标系中的表示
            rotation_matrix = np.array([
                [vec_x[0], vec_y[0], vec_z[0]],
                [vec_x[1], vec_y[1], vec_z[1]],
                [vec_x[2], vec_y[2], vec_z[2]]
            ])

            # 将旋转矩阵转换为四元数
            # tf_trans.quaternion_from_matrix 需要一个4x4的变换矩阵
            # 我们填充我们的3x3旋转矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3,:3] = rotation_matrix
            q = tf_trans.quaternion_from_matrix(transform_matrix)
            chessboard_orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            # --- 发布 PoseStamped 消息 ---
            pose_stamped_msg = PoseStamped()
            pose_stamped_msg.header.stamp = rospy.Time.now()
            pose_stamped_msg.header.frame_id = self.camera_frame_id # 使用从Aruco消息中获取的帧ID
            pose_stamped_msg.pose.position = chessboard_center
            pose_stamped_msg.pose.orientation = chessboard_orientation

            self.chessboard_pose_pub.publish(pose_stamped_msg)
            # rospy.loginfo_throttle(1.0, f"棋盘姿态已发布。中心: X:{center[0]:.3f} Y:{center[1]:.3f} Z:{center[2]:.3f}")

        elif len(detected_target_markers) > 0 :
            rospy.logwarn_throttle(2.0, f"未检测到所有4个角点标记。检测到的ID: {list(detected_target_markers.keys())}。需要 {self.corner_ids}。")
        else:
            rospy.logwarn_throttle(5.0, "未检测到用于棋盘定位的目标ArUco标记。")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        locator = ChessboardLocator()
        locator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("棋盘定位节点被中断。")
    except Exception as e:
        rospy.logerr(f"棋盘定位器发生未处理的异常: {e}")
        import traceback
        traceback.print_exc()