#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from std_msgs.msg import Header # 用于自定义消息中的header
# 导入你的自定义消息 - 请确保你的包名和消息文件名正确
# 例如: from your_package_name.msg import ChessboardCorners
from ee368_project.msg import ChessboardCorners # 使用你提供的包名

class ChessboardArucoCornersROS:
    def __init__(self):
        rospy.init_node('chessboard_aruco_corners_node', anonymous=False)

        # --- ArUco 参数 ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # ArUco标签的实际尺寸（米）
        aruco_dict_name_param = rospy.get_param("~aruco_dictionary_name", "DICT_6X6_250")

        # --- 棋盘角点对应的ArUco ID ---
        # 这些ID将用于从检测到的ArUco标签中挑选出代表棋盘四个角的标签
        self.corner_ids = {
            "top_left": rospy.get_param("~top_left_id", 0),
            "top_right": rospy.get_param("~top_right_id", 1),
            "bottom_left": rospy.get_param("~bottom_left_id", 2),
            "bottom_right": rospy.get_param("~bottom_right_id", 3)
        }
        self.expected_ids = set(self.corner_ids.values()) # 期望检测到的所有ID集合

        # --- 通用参数 ---
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_color_optical_frame")
        self.show_cv_window = rospy.get_param("~show_cv_window", True)

        # --- ArUco 字典和参数初始化 ---
        try:
            # 动态获取aruco字典的属性
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None: # getattr可能返回None而不是抛出异常
                raise AttributeError
        except AttributeError:
            rospy.logerr(f"无效的ArUco字典名称: {aruco_dict_name_param}. 将使用 DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250 # 默认值

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        try:
            self.parameters = aruco.DetectorParameters() # OpenCV 4.7+
        except AttributeError: # 兼容旧版OpenCV
            self.parameters = aruco.DetectorParameters_create()


        # --- RealSense 初始化 (只使用彩色流) ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = rospy.get_param("~color_width", 1280)
        self.color_height = rospy.get_param("~color_height", 720)
        self.fps = rospy.get_param("~fps", 30)

        # 只启用彩色流
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            rospy.logerr(f"启动RealSense管线失败: {e}")
            rospy.logerr("请检查相机是否连接，或者是否有其他程序正在使用相机。")
            rospy.signal_shutdown("RealSense管线启动失败。")
            return # 阻止后续代码执行如果管线启动失败

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [self.intrinsics_color.fx, 0, self.intrinsics_color.ppx],
            [0, self.intrinsics_color.fy, self.intrinsics_color.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array(self.intrinsics_color.coeffs, dtype=np.float32)
        # 处理畸变系数，确保格式正确
        if self.dist_coeffs is None or len(self.dist_coeffs) == 0 or self.intrinsics_color.model == rs.distortion.none:
            rospy.logwarn("相机报告无畸变或畸变系数为空，使用零畸变系数。")
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32) # OpenCV通常需要k1,k2,p1,p2,k3
        elif not (len(self.dist_coeffs) in [4, 5, 8, 12, 14]): # 检查常见长度
             rospy.logwarn(f"获取到非预期的畸变系数数量 ({len(self.dist_coeffs)}). 可能导致不准确的姿态估计。尝试使用前5个或零畸变。")
             # 如果长度不标准，最好用0，或者截取/填充到5
             if len(self.dist_coeffs) >= 5:
                 self.dist_coeffs = self.dist_coeffs.flatten()[:5].reshape(-1,1)
             else:
                 self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        if len(self.dist_coeffs.flatten()) < 5: # 确保至少有5个元素 (k1,k2,p1,p2,k3)
            rospy.logwarn(f"畸变系数少于5个 ({len(self.dist_coeffs.flatten())}), 用0填充到5个。")
            self.dist_coeffs = np.pad(self.dist_coeffs.flatten(), (0, 5 - len(self.dist_coeffs.flatten())), 'constant').reshape(-1,1)
        self.dist_coeffs = self.dist_coeffs.reshape(-1, 1) # 确保是列向量


        # --- 定义目标角点在标签局部坐标系中的坐标 (方案1) ---
        # 这是ArUco标签物理尺寸的一半
        self.s_half = self.marker_real_size_meters / 2.0
        # 目标角点：标签的左上角物理边界点。
        # 假设标签的局部坐标系原点在中心，X轴向右，Y轴向下（与OpenCV图像惯例匹配），Z轴向外。
        # 那么左上角是 (-s/2, -s/2, 0)
        self.target_corner_in_marker_coords = np.array([[-self.s_half], [self.s_half], [0.0]], dtype=np.float32)
        rospy.loginfo(f"目标角点在标签局部坐标系 (左上角物理边界): {self.target_corner_in_marker_coords.flatten()}")


        rospy.loginfo("RealSense相机内参 (fx, fy, cx, cy):")
        rospy.loginfo(f"  fx: {self.intrinsics_color.fx:.2f}, fy: {self.intrinsics_color.fy:.2f}")
        rospy.loginfo(f"  cx: {self.intrinsics_color.ppx:.2f}, cy: {self.intrinsics_color.ppy:.2f}")
        rospy.loginfo(f"畸变模型: {self.intrinsics_color.model}")
        rospy.loginfo(f"使用的畸变系数: {self.dist_coeffs.flatten()}")

        # --- ROS 发布器 ---
        # 发布计算得到的棋盘（由ArUco标签定义）的四个角点（每个角点是对应ArUco标签的左上角物理边界）
        self.chessboard_corners_pub = rospy.Publisher("/chessboard_corners", ChessboardCorners, queue_size=10)
        # 发布原始彩色图像，用于调试或其他节点使用
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)

        rospy.loginfo("ArUco棋盘角点检测节点已初始化。")
        rospy.loginfo(f"将发布棋盘角点到: /chessboard_corners")
        rospy.loginfo(f"将发布原始图像到: /camera/color/image_raw")
        rospy.on_shutdown(self.shutdown_hook) # 注册关闭时的清理函数

    def process_frame(self):
        try:
            # 等待一对连贯的帧: 彩色 (这里只有彩色)
            frames = self.pipeline.wait_for_frames(timeout_ms=1000) # 增加超时以防卡住
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"等待帧超时或错误: {e}")
            return

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            rospy.logwarn_throttle(5, "未接收到彩色帧")
            return

        # 将RealSense帧转换为OpenCV图像格式 (NumPy array)
        color_image = np.asanyarray(color_frame_rs.get_data())
        # 转换为灰度图进行ArUco检测
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测ArUco标签
        # corners: 检测到的标记的角点列表。对于每个标记，其角点按其原始顺序返回（通常是顺时针）。
        # ids: 每个检测到的标记的ID列表。
        # rejectedImgPoints: 包含被拒绝的候选标记的角点的列表。
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        current_time = rospy.Time.now()
        display_image = color_image.copy() # 创建图像副本用于绘制
        
        detected_corner_positions = {} # 存储每个期望ID对应的计算出的3D角点

        if ids is not None and len(ids) > 0: # 如果检测到任何ArUco标签
            # 在显示图像上绘制检测到的ArUco标签的轮廓和ID
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            # 估计每个检测到的ArUco标签的姿态
            # rvecs: 输出的旋转向量数组 (每个标签一个)
            # tvecs: 输出的平移向量数组 (每个标签一个)。tvec是标签坐标系原点在相机坐标系下的3D位置。
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id_arr in enumerate(ids): # 遍历所有检测到的标签
                marker_id = int(marker_id_arr[0]) # 获取当前标签的ID
                
                if marker_id in self.expected_ids: # 如果这个ID是我们期望的棋盘角ID之一
                    rvec = rvecs[i][0] # 当前标签的旋转向量 (3x1)
                    tvec = tvecs[i][0] # 当前标签的平移向量 (3x1) - 这是标签中心在相机坐标系的位置

                    # --- 方案1: 通过姿态估计推断标签的左上角物理边界点 ---
                    # 1. 将旋转向量 rvec 转换为旋转矩阵 R
                    R, _ = cv2.Rodrigues(rvec) # R 是 3x3 旋转矩阵

                    # 2. P_camera = R * P_marker_local + T_marker_center_to_camera
                    # self.target_corner_in_marker_coords 是目标角点在标签局部坐标系的位置 (3,1)
                    # R 是从标签局部坐标系到相机坐标系的旋转 (3,3)
                    # tvec 需要 reshape 成 (3,1) 列向量 (标签中心在相机系的位置)
                    
                    tvec_col = tvec.reshape(3, 1) # 确保tvec是列向量
                    
                    # 将局部角点旋转到与相机坐标系对齐的方向
                    corner_in_camera_coords_rotated = np.dot(R, self.target_corner_in_marker_coords)
                    
                    # 加上标签中心的平移向量，得到角点在相机坐标系下的最终位置
                    corner_in_camera_coords = corner_in_camera_coords_rotated + tvec_col
                    
                    # corner_in_camera_coords 现在是 (3,1) 的 NumPy 数组，包含 (x,y,z)
                    x_cam = corner_in_camera_coords[0, 0]
                    y_cam = corner_in_camera_coords[1, 0]
                    z_cam = corner_in_camera_coords[2, 0]
                    # --- 结束方案1 ---
                                     
                    point_3d = Point(x=x_cam, y=y_cam, z=z_cam)
                    detected_corner_positions[marker_id] = point_3d # 存储计算得到的3D点
                    
                    # 在显示图像上绘制每个ArUco标签的坐标轴
                    try:
                        cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                        # 可选：绘制计算出的角点在图像上的投影以进行验证
                        if z_cam > 0: # 确保点在相机前方
                             # 简单针孔相机模型投影 (忽略畸变，用于快速可视化)
                             u_proj = int(self.intrinsics_color.fx * x_cam / z_cam + self.intrinsics_color.ppx)
                             v_proj = int(self.intrinsics_color.fy * y_cam / z_cam + self.intrinsics_color.ppy)
                             # 确保投影点在图像范围内
                             if 0 <= u_proj < self.color_width and 0 <= v_proj < self.color_height:
                                 cv2.circle(display_image, (u_proj, v_proj), 7, (0, 255, 255), -1) # 黄色大圆点
                    except AttributeError: # 兼容旧版OpenCV
                        aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
        
        # 检查是否所有期望的棋盘角ArUco标签都被检测到并计算了其对应的3D点
        if len(detected_corner_positions) == len(self.expected_ids):
            corners_msg = ChessboardCorners() # 创建自定义消息对象
            corners_msg.header.stamp = current_time
            corners_msg.header.frame_id = self.camera_frame_id
            
            # 填充消息字段
            corners_msg.top_left = detected_corner_positions[self.corner_ids["top_left"]]
            corners_msg.top_right = detected_corner_positions[self.corner_ids["top_right"]]
            corners_msg.bottom_left = detected_corner_positions[self.corner_ids["bottom_left"]]
            corners_msg.bottom_right = detected_corner_positions[self.corner_ids["bottom_right"]]
            
            self.chessboard_corners_pub.publish(corners_msg) # 发布消息
            rospy.loginfo_throttle(1, "成功发布棋盘角点 (ArUco - 左上角物理点)")
        elif ids is not None and len(ids) > 0: # 检测到一些ArUco，但不是全部期望的
            rospy.logwarn_throttle(1, f"检测到ArUco标签，但未集齐所有期望的棋盘角点。已检测ID: {list(detected_corner_positions.keys())}, 期望ID: {self.expected_ids}")
        else: # 没有检测到任何ArUco标签
            rospy.logwarn_throttle(5, "未检测到ArUco标签。")

        # --- 发布原始彩色图像 ---
        ros_image_msg = Image()
        ros_image_msg.header.stamp = current_time # 使用与角点消息相同的时间戳
        ros_image_msg.header.frame_id = self.camera_frame_id
            
        ros_image_msg.height = color_image.shape[0]
        ros_image_msg.width = color_image.shape[1]
        ros_image_msg.encoding = "bgr8" # RealSense配置为BGR8
        ros_image_msg.is_bigendian = 0 # 通常为小端
        ros_image_msg.step = color_image.shape[1] * 3 # 每行的字节数: width * num_channels * bytes_per_channel
        ros_image_msg.data = color_image.tobytes() # 图像数据展平为字节串
        self.image_pub.publish(ros_image_msg)

        # --- 显示OpenCV窗口 (如果启用) ---
        if self.show_cv_window:
            cv2.imshow("ArUco Chessboard Corner Detection", display_image)
            key = cv2.waitKey(1) & 0xFF # 等待按键，取低8位以兼容不同平台
            if key == 27 or key == ord('q'): # ESC键或q键退出
                rospy.signal_shutdown("ESC or q key pressed in CV window")


    def run(self):
        # 设置循环频率，可以与相机FPS大致匹配，但主要由wait_for_frames控制
        rate = rospy.Rate(self.fps if self.fps > 0 else 30) 
        while not rospy.is_shutdown():
            self.process_frame()
            try:
                rate.sleep()
            except rospy.ROSInterruptException: # 被Ctrl+C中断
                rospy.loginfo("ROS 中断 (rate.sleep)。正在关闭。")
                break
            except rospy.ROSTimeMovedBackwardsError: # 时间回拨错误
                rospy.logwarn("ROS time moved backwards. Skipping sleep.")
        
        # 循环结束后，如果OpenCV窗口还开着，确保关闭
        if self.show_cv_window:
            cv2.destroyAllWindows()


    def shutdown_hook(self):
        """ROS关闭时调用的清理函数"""
        rospy.loginfo("ArUco检测节点正在关闭...")
        if hasattr(self, 'pipeline') and self.pipeline:
            rospy.loginfo("正在停止RealSense管线...")
            try:
                # 检查profile是否存在且有效，避免在pipeline未成功启动或已停止时出错
                if hasattr(self, 'profile') and self.profile and self.profile.get_device():
                     self.pipeline.stop()
                     rospy.loginfo("RealSense管线已停止。")
                else:
                    rospy.logwarn("RealSense管线profile无效或未初始化，可能未启动或已被停止。跳过显式停止。")
            except RuntimeError as e:
                rospy.logwarn(f"停止RealSense管线时发生错误 (可能已被停止): {e}")

        if self.show_cv_window:
            cv2.destroyAllWindows() # 关闭所有OpenCV窗口
        rospy.loginfo("ArUco检测节点关闭完成。")


if __name__ == '__main__':
    try:
        detector = ChessboardArucoCornersROS()
        # 确保 RealSense 管线成功初始化后再运行主循环
        if hasattr(detector, 'profile') and detector.profile:
            detector.run()
        else:
            # 如果profile不存在，说明__init__中RealSense启动失败并已调用rospy.signal_shutdown
            rospy.logerr("RealSense管线未成功初始化，节点将不运行主处理循环。")
    except rospy.ROSInterruptException:
        rospy.loginfo("ArUco棋盘角点检测节点被中断 (main)。")
    except Exception as e:
        rospy.logfatal(f"ArUco棋盘角点检测器出现未处理的严重异常: {e}")
        import traceback
        traceback.print_exc() # 打印详细的堆栈跟踪
    finally:
        # 确保在任何退出情况下（包括异常）都尝试关闭OpenCV窗口
        if 'cv2' in globals() and ChessboardArucoCornersROS.show_cv_window: # 检查cv2是否已导入且窗口应显示
            cv2.destroyAllWindows()
        rospy.loginfo("ArUco节点最终退出。")