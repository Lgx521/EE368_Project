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
from ee368_project.msg import ChessboardCorners # 使用你提供的包名
from ee368_project.msg import ChessboardPixelCorners # 新增：导入像素角点消息

class ChessboardArucoCornersROS:
    def __init__(self):
        rospy.init_node('chessboard_aruco_corners_node', anonymous=False)

        # --- ArUco 参数 ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # ArUco标签的实际尺寸（米）
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

        # --- ArUco 字典和参数初始化 ---
        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None:
                raise AttributeError
        except AttributeError:
            rospy.logerr(f"无效的ArUco字典名称: {aruco_dict_name_param}. 将使用 DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create()


        # --- RealSense 初始化 (只使用彩色流) ---
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
            rospy.logerr("请检查相机是否连接，或者是否有其他程序正在使用相机。")
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
             rospy.logwarn(f"获取到非预期的畸变系数数量 ({len(self.dist_coeffs)}). 可能导致不准确的姿态估计。尝试使用前5个或零畸变。")
             if len(self.dist_coeffs) >= 5:
                 self.dist_coeffs = self.dist_coeffs.flatten()[:5].reshape(-1,1)
             else:
                 self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        if len(self.dist_coeffs.flatten()) < 5:
            rospy.logwarn(f"畸变系数少于5个 ({len(self.dist_coeffs.flatten())}), 用0填充到5个。")
            self.dist_coeffs = np.pad(self.dist_coeffs.flatten(), (0, 5 - len(self.dist_coeffs.flatten())), 'constant').reshape(-1,1)
        self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        # --- 定义目标角点在标签局部坐标系中的坐标 ---
        # 这是ArUco标签物理尺寸的一半
        self.s_half = self.marker_real_size_meters / 2.0
        # 目标角点：标签的左上角物理边界点。
        # 假设标签的局部坐标系原点在中心，X轴向右，Y轴向下（与OpenCV图像惯例匹配），Z轴向外。
        # 那么左上角是 (-s/2, -s/2, 0)
        # MODIFIED: Corrected to truly represent top-left corner (-s/2, -s/2, 0)
        self.target_corner_in_marker_coords = np.array([[-self.s_half], [self.s_half], [0.0]], dtype=np.float32)
        rospy.loginfo(f"目标3D角点在标签局部坐标系 (左上角物理边界, X右Y下Z外): {self.target_corner_in_marker_coords.flatten()}")

        rospy.loginfo("RealSense相机内参 (fx, fy, cx, cy):")
        rospy.loginfo(f"  fx: {self.intrinsics_color.fx:.2f}, fy: {self.intrinsics_color.fy:.2f}")
        rospy.loginfo(f"  cx: {self.intrinsics_color.ppx:.2f}, cy: {self.intrinsics_color.ppy:.2f}")
        rospy.loginfo(f"畸变模型: {self.intrinsics_color.model}")
        rospy.loginfo(f"使用的畸变系数: {self.dist_coeffs.flatten()}")

        # --- ROS 发布器 ---
        # 发布计算得到的棋盘（由ArUco标签定义）的四个角点（每个角点是对应ArUco标签的左上角物理边界）的3D坐标
        self.chessboard_corners_pub = rospy.Publisher("/chessboard_corners", ChessboardCorners, queue_size=10)
        # 新增：发布棋盘角点（ArUco标签的左上角）的2D像素坐标
        self.chessboard_pixel_corners_pub = rospy.Publisher("/chessboard_pixel_corners", ChessboardPixelCorners, queue_size=10)
        # 发布原始彩色图像，用于调试或其他节点使用
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)

        rospy.loginfo("ArUco棋盘角点检测节点已初始化。")
        rospy.loginfo(f"将发布棋盘3D角点到: /chessboard_corners")
        rospy.loginfo(f"将发布棋盘2D像素角点到: /chessboard_pixel_corners") # 新增日志
        rospy.loginfo(f"将发布原始图像到: /camera/color/image_raw")
        rospy.on_shutdown(self.shutdown_hook)

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"等待帧超时或错误: {e}")
            return

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            rospy.logwarn_throttle(5, "未接收到彩色帧")
            return

        color_image = np.asanyarray(color_frame_rs.get_data())
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # corners: list of np.array (1, 4, 2) for each marker. Corners are (x,y) pixels.
        # Order: top-left, top-right, bottom-right, bottom-left
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        current_time = rospy.Time.now()
        display_image = color_image.copy()
        
        detected_3d_corner_positions = {} # 存储每个期望ID对应的计算出的3D角点
        detected_2d_pixel_positions = {}  # 新增: 存储每个期望ID对应的检测到的2D像素角点

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                
                if marker_id in self.expected_ids:
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    
                    # --- 计算3D角点 (左上角物理边界) ---
                    R, _ = cv2.Rodrigues(rvec)
                    tvec_col = tvec.reshape(3, 1)
                    corner_in_camera_coords_rotated = np.dot(R, self.target_corner_in_marker_coords)
                    corner_in_camera_coords = corner_in_camera_coords_rotated + tvec_col
                    
                    x_cam = corner_in_camera_coords[0, 0]
                    y_cam = corner_in_camera_coords[1, 0]
                    z_cam = corner_in_camera_coords[2, 0]
                                     
                    point_3d = Point(x=x_cam, y=y_cam, z=z_cam)
                    detected_3d_corner_positions[marker_id] = point_3d
                    
                    # --- 获取2D像素角点 (左上角检测点) ---
                    # corners[i] is (1,4,2). corners[i][0] is (4,2) with TL, TR, BR, BL points.
                    # corners[i][0][0] is the top-left pixel [u,v] of the i-th marker.
                    marker_top_left_pixel = corners[i][0][0] 
                    u_px = float(marker_top_left_pixel[0])
                    v_px = float(marker_top_left_pixel[1])
                    pixel_point_2d = Point(x=u_px, y=v_px, z=0.0) # Using Point msg, z=0 for 2D
                    detected_2d_pixel_positions[marker_id] = pixel_point_2d

                    try:
                        cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                        if z_cam > 0:
                             u_proj = int(self.intrinsics_color.fx * x_cam / z_cam + self.intrinsics_color.ppx)
                             v_proj = int(self.intrinsics_color.fy * y_cam / z_cam + self.intrinsics_color.ppy)
                             if 0 <= u_proj < self.color_width and 0 <= v_proj < self.color_height:
                                 cv2.circle(display_image, (u_proj, v_proj), 7, (0, 255, 255), -1) # Yellow dot for projected 3D point
                                 # Draw the detected 2D pixel point for comparison
                                 cv2.circle(display_image, (int(u_px), int(v_px)), 5, (255, 0, 255), -1) # Magenta dot for detected 2D point
                    except AttributeError:
                        aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
        
        # 检查是否所有期望的棋盘角ArUco标签都被检测到
        if len(detected_3d_corner_positions) == len(self.expected_ids) and \
           len(detected_2d_pixel_positions) == len(self.expected_ids): # Ensure both dicts are fully populated
            
            # --- 发布3D角点 ---
            corners_msg_3d = ChessboardCorners()
            corners_msg_3d.header.stamp = current_time
            corners_msg_3d.header.frame_id = self.camera_frame_id
            
            corners_msg_3d.top_left = detected_3d_corner_positions[self.corner_ids["top_left"]]
            corners_msg_3d.top_right = detected_3d_corner_positions[self.corner_ids["top_right"]]
            corners_msg_3d.bottom_left = detected_3d_corner_positions[self.corner_ids["bottom_left"]]
            corners_msg_3d.bottom_right = detected_3d_corner_positions[self.corner_ids["bottom_right"]]
            
            self.chessboard_corners_pub.publish(corners_msg_3d)
            rospy.loginfo_throttle(1, "成功发布棋盘3D角点 (ArUco - 左上角物理点)")

            # --- 新增：发布2D像素角点 ---
            corners_msg_2d_pixel = ChessboardPixelCorners()
            corners_msg_2d_pixel.header.stamp = current_time
            corners_msg_2d_pixel.header.frame_id = self.camera_frame_id

            corners_msg_2d_pixel.top_left_px = detected_2d_pixel_positions[self.corner_ids["top_left"]]
            corners_msg_2d_pixel.top_right_px = detected_2d_pixel_positions[self.corner_ids["top_right"]]
            corners_msg_2d_pixel.bottom_left_px = detected_2d_pixel_positions[self.corner_ids["bottom_left"]]
            corners_msg_2d_pixel.bottom_right_px = detected_2d_pixel_positions[self.corner_ids["bottom_right"]]
            
            self.chessboard_pixel_corners_pub.publish(corners_msg_2d_pixel)
            rospy.loginfo_throttle(1, "成功发布棋盘2D像素角点 (ArUco - 左上角检测点)")

        elif ids is not None and len(ids) > 0:
            # Log which specific type of corners are missing if only one set is incomplete, or general if both
            missing_3d = len(self.expected_ids) - len(detected_3d_corner_positions)
            missing_2d = len(self.expected_ids) - len(detected_2d_pixel_positions)
            log_msg = f"检测到ArUco标签，但未集齐所有期望的棋盘角点。"
            if missing_3d > 0 :
                log_msg += f" 3D点所需ID: {list(detected_3d_corner_positions.keys())} (缺{missing_3d}个)."
            if missing_2d > 0 :
                 log_msg += f" 2D点所需ID: {list(detected_2d_pixel_positions.keys())} (缺{missing_2d}个)."
            log_msg += f" 期望ID: {self.expected_ids}"
            rospy.logwarn_throttle(1, log_msg)
        else:
            rospy.logwarn_throttle(5, "未检测到ArUco标签。")

        # --- 发布原始彩色图像 ---
        ros_image_msg = Image()
        ros_image_msg.header.stamp = current_time
        ros_image_msg.header.frame_id = self.camera_frame_id
            
        ros_image_msg.height = color_image.shape[0]
        ros_image_msg.width = color_image.shape[1]
        ros_image_msg.encoding = "bgr8"
        ros_image_msg.is_bigendian = 0
        ros_image_msg.step = color_image.shape[1] * 3
        ros_image_msg.data = color_image.tobytes()
        self.image_pub.publish(ros_image_msg)

        if self.show_cv_window:
            cv2.imshow("ArUco Chessboard Corner Detection", display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                rospy.signal_shutdown("ESC or q key pressed in CV window")


    def run(self):
        rate = rospy.Rate(self.fps if self.fps > 0 else 30) 
        while not rospy.is_shutdown():
            self.process_frame()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS 中断 (rate.sleep)。正在关闭。")
                break
            except rospy.ROSTimeMovedBackwardsError:
                rospy.logwarn("ROS time moved backwards. Skipping sleep.")
        
        if self.show_cv_window:
            cv2.destroyAllWindows()


    def shutdown_hook(self):
        rospy.loginfo("ArUco检测节点正在关闭...")
        if hasattr(self, 'pipeline') and self.pipeline:
            rospy.loginfo("正在停止RealSense管线...")
            try:
                if hasattr(self, 'profile') and self.profile and self.profile.get_device():
                     self.pipeline.stop()
                     rospy.loginfo("RealSense管线已停止。")
                else:
                    rospy.logwarn("RealSense管线profile无效或未初始化，可能未启动或已被停止。跳过显式停止。")
            except RuntimeError as e:
                rospy.logwarn(f"停止RealSense管线时发生错误 (可能已被停止): {e}")

        if self.show_cv_window:
            cv2.destroyAllWindows()
        rospy.loginfo("ArUco检测节点关闭完成。")


if __name__ == '__main__':
    try:
        detector = ChessboardArucoCornersROS()
        if hasattr(detector, 'profile') and detector.profile:
            detector.run()
        else:
            rospy.logerr("RealSense管线未成功初始化，节点将不运行主处理循环。")
    except rospy.ROSInterruptException:
        rospy.loginfo("ArUco棋盘角点检测节点被中断 (main)。")
    except Exception as e:
        rospy.logfatal(f"ArUco棋盘角点检测器出现未处理的严重异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Check if 'cv2' is in globals and if 'show_cv_window' was intended to be true for the class instance
        # This is a bit tricky because detector might not be fully initialized if __init__ failed early
        # A class attribute might be safer if show_cv_window parameter is always the same.
        # For now, we assume cv2 is imported if the class definition was reached.
        # And rely on the instance's show_cv_window if detector was created.
        try:
            if detector and detector.show_cv_window: # Check if detector exists and its show_cv_window is True
                 cv2.destroyAllWindows()
            elif 'cv2' in globals() and ChessboardArucoCornersROS.show_cv_window: # Fallback for very early errors
                 cv2.destroyAllWindows()
        except NameError: # detector might not be defined
            if 'cv2' in globals() and ChessboardArucoCornersROS.show_cv_window: # Check class default if instance not available
                cv2.destroyAllWindows()
        except AttributeError: # detector might exist but not have show_cv_window yet
            if 'cv2' in globals() and ChessboardArucoCornersROS.show_cv_window:
                cv2.destroyAllWindows()

        rospy.loginfo("ArUco节点最终退出。")