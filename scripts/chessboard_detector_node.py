#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import Point
from std_msgs.msg import Header # 用于自定义消息中的header
# 导入你的自定义消息
from ee368_project.msg import ChessboardCorners # 确保包名正确

class ChessboardArucoCornersROS:
    def __init__(self):
        rospy.init_node('chessboard_aruco_corners_node', anonymous=False)

        # --- ArUco 参数 ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # ArUco标签的实际尺寸（米）
        # 将默认字典修改为 DICT_6X6_250
        aruco_dict_name_param = rospy.get_param("~aruco_dictionary_name", "DICT_6X6_250")

        # --- 棋盘角点对应的ArUco ID ---
        self.corner_ids = {
            "top_left": rospy.get_param("~top_left_id", 0),
            "top_right": rospy.get_param("~top_right_id", 1),
            "bottom_left": rospy.get_param("~bottom_left_id", 2),
            "bottom_right": rospy.get_param("~bottom_right_id", 3)
        }
        self.expected_ids = set(self.corner_ids.values())

        # --- 通用参数 ---
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_color_optical_frame")
        self.show_cv_window = rospy.get_param("~show_cv_window", True)

        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None:
                raise AttributeError
        except AttributeError:
            rospy.logerr(f"无效的ArUco字典名称: {aruco_dict_name_param}. 将使用 DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250 # 默认值

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError: # For older OpenCV versions
            self.parameters = aruco.DetectorParameters_create()


        # --- RealSense 初始化 ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = rospy.get_param("~color_width", 1280)
        self.color_height = rospy.get_param("~color_height", 720)
        self.fps = rospy.get_param("~fps", 30)

        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            rospy.logerr(f"启动RealSense管线失败: {e}")
            rospy.signal_shutdown("RealSense管线启动失败。")
            return

        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [self.intrinsics_color.fx, 0, self.intrinsics_color.ppx],
            [0, self.intrinsics_color.fy, self.intrinsics_color.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array(self.intrinsics_color.coeffs, dtype=np.float32)
        if self.dist_coeffs is None or len(self.dist_coeffs) == 0 or self.intrinsics_color.model == rs.distortion.none:
            rospy.logwarn("相机报告无畸变或畸变系数为空，使用零畸变系数。")
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        elif not (len(self.dist_coeffs) in [4, 5, 8, 12, 14]):
             rospy.logwarn(f"获取到非预期的畸变系数数量 ({len(self.dist_coeffs)}). 可能导致不准确的姿态估计。使用零畸变。")
             self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        if len(self.dist_coeffs) < 5: # 确保至少有5个元素
            rospy.logwarn(f"畸变系数少于5个 ({len(self.dist_coeffs)}), 用0填充到5个。")
            self.dist_coeffs = np.pad(self.dist_coeffs.flatten(), (0, 5 - len(self.dist_coeffs.flatten())), 'constant').reshape(-1,1)


        rospy.loginfo("RealSense相机内参 (fx, fy, cx, cy):")
        rospy.loginfo(f"  fx: {self.intrinsics_color.fx:.2f}, fy: {self.intrinsics_color.fy:.2f}")
        rospy.loginfo(f"  cx: {self.intrinsics_color.ppx:.2f}, cy: {self.intrinsics_color.ppy:.2f}")
        rospy.loginfo(f"畸变系数: {self.dist_coeffs.flatten()}")

        # --- ROS 发布器 ---
        self.chessboard_corners_pub = rospy.Publisher("/chessboard_corners", ChessboardCorners, queue_size=10)

        rospy.loginfo("ArUco棋盘角点检测节点已初始化。发布到 /chessboard_corners")
        rospy.on_shutdown(self.shutdown_hook)

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"等待帧超时: {e}")
            return

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            rospy.logwarn_throttle(5, "未接收到彩色帧")
            return

        color_image = np.asanyarray(color_frame_rs.get_data())
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 修改: 从 detectMarkers 调用中移除 cameraMatrix 和 distCoeff
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        current_time = rospy.Time.now()
        display_image = color_image.copy() 
        
        detected_corner_positions = {} # 用于存储检测到的角点3D坐标

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            # estimatePoseSingleMarkers 返回每个标签的旋转向量(rvec)和平移向量(tvec)
            # tvec 是标签坐标系原点在相机坐标系下的3D坐标
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0]) # ids 是 (N, 1) 的数组
                
                if marker_id in self.expected_ids:
                    tvec = tvecs[i][0] # tvecs[i] 是一个 (1,3) 的数组，取其第一个（也是唯一一个）元素
                    
                    point_3d = Point(x=tvec[0], y=tvec[1], z=tvec[2])
                    detected_corner_positions[marker_id] = point_3d
                    
                    try:
                        cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_real_size_meters / 2)
                    except AttributeError: # for older OpenCV
                        aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_real_size_meters / 2)
        
        if len(detected_corner_positions) == len(self.expected_ids):
            corners_msg = ChessboardCorners()
            corners_msg.header.stamp = current_time
            corners_msg.header.frame_id = self.camera_frame_id
            
            corners_msg.top_left = detected_corner_positions[self.corner_ids["top_left"]]
            corners_msg.top_right = detected_corner_positions[self.corner_ids["top_right"]]
            corners_msg.bottom_left = detected_corner_positions[self.corner_ids["bottom_left"]]
            corners_msg.bottom_right = detected_corner_positions[self.corner_ids["bottom_right"]]
            
            self.chessboard_corners_pub.publish(corners_msg)
            rospy.loginfo_throttle(1, "成功发布棋盘角点 (ArUco)")
        elif ids is not None and len(ids) > 0:
            rospy.logwarn_throttle(1, f"检测到ArUco标签，但未集齐所有棋盘角点。已检测: {list(detected_corner_positions.keys())}")
        else:
            rospy.logwarn_throttle(5, "未检测到ArUco标签。")


        if self.show_cv_window:
            cv2.imshow("ArUco Chessboard Corner Detection", display_image)
            key = cv2.waitKey(1)
            if key == 27: # ESC 键
                rospy.signal_shutdown("ESC key pressed")


    def run(self):
        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            self.process_frame()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS 中断。正在关闭。")
                break
        
        if self.show_cv_window:
            cv2.destroyAllWindows()


    def shutdown_hook(self):
        rospy.loginfo("正在停止RealSense管线...")
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if self.show_cv_window:
            cv2.destroyAllWindows()
        rospy.loginfo("关闭完成。")


if __name__ == '__main__':
    try:
        detector = ChessboardArucoCornersROS()
        if hasattr(detector, 'profile') and detector.profile:
            detector.run()
        else:
            rospy.logerr("RealSense管线未成功初始化，节点退出。")
    except rospy.ROSInterruptException:
        rospy.loginfo("ArUco棋盘角点检测节点被中断。")
    except Exception as e:
        rospy.logerr(f"ArUco棋盘角点检测器出现未处理异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rospy.is_shutdown() and hasattr(rospy, 'get_param') and rospy.get_param("~show_cv_window", False):
             cv2.destroyAllWindows()