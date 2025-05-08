#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import rospy

# 导入自定义消息和必要的 geometry_msgs
# 修改 'ee368_project' 为你实际的包名
from ee368_project.msg import ArucoMarker, ArucoMarkerArray
from geometry_msgs.msg import Pose, Quaternion # Pose for position and orientation
from std_msgs.msg import Header # For message headers
import tf.transformations as tf_trans # For converting rotation vector to quaternion

class ArucoRealSenseDetector:
    def __init__(self):
        # 初始化 ROS 节点，节点名称为 aruco_realsense_direct_detector
        rospy.init_node('aruco_realsense_direct_detector', anonymous=False)

        # --- 从参数服务器获取参数 ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # ArUco 标记的实际边长（米）
        aruco_dict_name_param = rospy.get_param("~aruco_dictionary_name", "DICT_6X6_250") # ArUco 字典名称
        self.show_cv_window = rospy.get_param("~show_cv_window", True) # 是否显示 OpenCV 调试窗口
        self.publish_topic_name = rospy.get_param("~publish_topic", "aruco_poses") # 发布标记位姿的话题名称
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_color_optical_frame") # 用于消息头的坐标系 ID

        # --- ArUco 初始化 ---
        try:
            self.aruco_dictionary_name_cv = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name_cv is None: # 防御性检查
                rospy.logerr(f"Failed to get attribute for dictionary: {aruco_dict_name_param}. Falling back.")
                raise AttributeError 
        except AttributeError:
            rospy.logwarn(f"Invalid ArUco dictionary name: {aruco_dict_name_param} from param server. Using default DICT_6X6_250.")
            self.aruco_dictionary_name_cv = aruco.DICT_6X6_250
        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name_cv)
        
        # 初始化 ArUco 检测器参数 (兼容旧版 OpenCV)
        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create()
        # self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX # 可选：角点精细化以提高精度

        # --- RealSense 初始化 ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = rospy.get_param("~color_width", 1280)
        self.color_height = rospy.get_param("~color_height", 720)
        self.color_fps = rospy.get_param("~color_fps", 30)

        # 配置并启动 RealSense 彩色数据流
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
            rospy.loginfo("RealSense pipeline started successfully.")
        except RuntimeError as e:
            rospy.logerr(f"Failed to start RealSense pipeline: {e}")
            rospy.signal_shutdown("RealSense pipeline failed to start. Shutting down node.")
            return # 关键：如果 pipeline 启动失败，则退出 __init__

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [self.intrinsics_color.fx, 0, self.intrinsics_color.ppx],
            [0, self.intrinsics_color.fy, self.intrinsics_color.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array(self.intrinsics_color.coeffs, dtype=np.float32)
        # 对畸变系数进行有效性检查和处理
        if self.dist_coeffs is None or len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32) # 默认为5个0 (无畸变)
            rospy.logwarn("Distortion coefficients not found or empty from RealSense, assuming zero distortion (k1=k2=p1=p2=k3=0).")
        elif not (len(self.dist_coeffs.flatten()) in [4, 5, 8, 12, 14]):
             # OpenCV estimatePoseSingleMarkers 通常期望 4, 5, 8, 12, 或 14 个畸变系数
            rospy.logwarn(f"Unexpected number of distortion coefficients ({len(self.dist_coeffs.flatten())}) from RealSense. "
                          f"Using zero distortion for pose estimation. Coeffs: {self.dist_coeffs.flatten()}")
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        # 如果长度不为5但属于可接受范围，确保它是(N,1)或(1,N)的形状
        self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)


        # --- ROS Publisher 初始化 ---
        self.marker_array_pub = rospy.Publisher(self.publish_topic_name, ArucoMarkerArray, queue_size=10)

        rospy.loginfo(f"ArUco RealSense detector node '{rospy.get_name()}' initialized.")
        rospy.loginfo(f"Publishing ArUco marker poses to topic: '{self.publish_topic_name}'")
        rospy.loginfo(f"Marker real size: {self.marker_real_size_meters} m")
        rospy.loginfo(f"Using ArUco dictionary: {aruco_dict_name_param}")
        rospy.loginfo(f"Camera frame ID for published messages: '{self.camera_frame_id}'")

        # 注册关闭时的回调函数
        rospy.on_shutdown(self.shutdown_hook)

    def rvec_tvec_to_pose_msg(self, rvec, tvec):
        """
        Converts a rotation vector (rvec) and translation vector (tvec)
        from OpenCV to a geometry_msgs/Pose message.
        """
        pose_msg = Pose()
        pose_msg.position.x = tvec[0]
        pose_msg.position.y = tvec[1]
        pose_msg.position.z = tvec[2]
        
        # Convert rotation vector (Rodrigues vector) to quaternion
        rotation_matrix, _ = cv2.Rodrigues(rvec) # Convert rvec to 3x3 rotation matrix
        
        # tf.transformations.quaternion_from_matrix expects a 4x4 homogeneous matrix
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix # Place 3x3 rotation matrix in top-left
        
        quaternion = tf_trans.quaternion_from_matrix(homogeneous_matrix)
        pose_msg.orientation.x = quaternion[0]
        pose_msg.orientation.y = quaternion[1]
        pose_msg.orientation.z = quaternion[2]
        pose_msg.orientation.w = quaternion[3]
        
        return pose_msg

    def process_frames_and_detect(self):
        """
        Gets frames from RealSense, detects ArUco markers, estimates their poses,
        and publishes them. Also handles optional OpenCV window display.
        Returns the color image for display, or None if an error occurs.
        """
        try:
            # 等待一对连贯的帧：深度和颜色
            frames = self.pipeline.wait_for_frames(timeout_ms=1000) # 1秒超时
        except RuntimeError as e:
            rospy.logwarn_throttle(5.0, f"Timeout or error waiting for RealSense frames: {e}")
            return None # 如果获取帧失败，则返回 None

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            rospy.logwarn_throttle(5.0, "No color frame received from RealSense.")
            return None

        # 将 RealSense 图像数据转换为 OpenCV 格式 (NumPy array)
        color_image_cv = np.asanyarray(color_frame_rs.get_data())
        # 将 BGR 图像转换为灰度图，用于 ArUco 检测
        gray_image_cv = cv2.cvtColor(color_image_cv, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco 标记
        corners, ids, rejected_img_points = aruco.detectMarkers(
            gray_image_cv, self.dictionary, parameters=self.parameters
        )

        # 获取当前时间，用于消息头
        current_time = rospy.Time.now()
        
        # 创建 ArucoMarkerArray 消息
        marker_array_msg = ArucoMarkerArray()
        marker_array_msg.header.stamp = current_time
        marker_array_msg.header.frame_id = self.camera_frame_id # 所有位姿都在这个相机坐标系下

        # 如果检测到任何标记
        if ids is not None and len(ids) > 0:
            rospy.logdebug(f"Detected {len(ids)} ArUco IDs: {ids.flatten().tolist()}")
            # 估计每个标记的位姿
            # rvecs: 旋转向量列表, tvecs: 平移向量列表
            rvecs, tvecs, _obj_points = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            # 遍历所有检测到的标记
            for i in range(len(ids)):
                marker_msg = ArucoMarker()
                marker_msg.header.stamp = current_time
                marker_msg.header.frame_id = self.camera_frame_id # 单个标记的位姿也在此坐标系
                
                marker_msg.id = int(ids[i][0]) # 获取标记 ID
                
                # 获取当前标记的平移向量和旋转向量
                tvec_camera_frame = tvecs[i][0] # Shape (3,)
                rvec_camera_frame = rvecs[i][0] # Shape (3,)
                
                # 将 rvec 和 tvec 转换为 geometry_msgs/Pose
                marker_msg.pose_camera = self.rvec_tvec_to_pose_msg(rvec_camera_frame, tvec_camera_frame)
                
                marker_array_msg.markers.append(marker_msg)
                
                # 可选：在 OpenCV 窗口中绘制检测结果
                if self.show_cv_window:
                    # 只绘制当前处理的标记的边界框和 ID
                    aruco.drawDetectedMarkers(color_image_cv, [corners[i]], np.array([[ids[i][0]]]))
                    # 绘制坐标轴 (兼容旧版 OpenCV)
                    try:
                        cv2.drawFrameAxes(color_image_cv, self.camera_matrix, self.dist_coeffs, 
                                          rvec_camera_frame, tvec_camera_frame, self.marker_real_size_meters / 2.0)
                    except AttributeError:
                        aruco.drawAxis(color_image_cv, self.camera_matrix, self.dist_coeffs, 
                                       rvec_camera_frame, tvec_camera_frame, self.marker_real_size_meters / 2.0)
            
            # 如果 markers 列表不为空 (即成功处理了至少一个标记)，则发布消息
            if marker_array_msg.markers:
                self.marker_array_pub.publish(marker_array_msg)
                rospy.logdebug(f"Published {len(marker_array_msg.markers)} ArUco marker poses.")
        else:
            rospy.logdebug("No ArUco markers detected in the current frame.")
        
        return color_image_cv # 返回处理后的图像，用于可选的 OpenCV 显示

    def run(self):
        rospy.loginfo(f"Node '{rospy.get_name()}' main loop started. Press Ctrl+C to exit.")
        # 设置循环频率，尽量与相机帧率匹配，或略低以避免不必要的 CPU 占用
        loop_rate = rospy.Rate(self.color_fps if self.color_fps > 0 else 10) # 如果fps为0，则默认为10Hz

        while not rospy.is_shutdown():
            # 处理帧并进行检测 (此方法内部会发布消息)
            processed_display_image = self.process_frames_and_detect()

            # 如果启用了 OpenCV 窗口并且成功获取了图像
            if self.show_cv_window and processed_display_image is not None:
                cv2.imshow("ArUco Detection - PyRealSense Pose Publisher", processed_display_image)
                key = cv2.waitKey(1) & 0xFF # 等待1ms，处理窗口事件
                if key == ord('q'): # 如果按下 'q' 键
                    rospy.loginfo("'q' key pressed in OpenCV window. Shutting down node.")
                    rospy.signal_shutdown("'q' pressed by user.") # 请求 ROS 关闭
                    break # 退出循环

            try:
                loop_rate.sleep() # 维持循环频率
            except rospy.ROSInterruptException: # 在 sleep 时捕获 Ctrl+C
                rospy.loginfo("ROS sleep interrupted. Node is shutting down.")
                break # 退出循环

    def shutdown_hook(self):
        """ROS shutdown hook."""
        rospy.loginfo(f"Shutdown hook called for node '{rospy.get_name()}'. Cleaning up...")
        if hasattr(self, 'pipeline') and self.pipeline: # 确保 pipeline 已初始化
            try:
                rospy.loginfo("Attempting to stop RealSense pipeline...")
                self.pipeline.stop()
                rospy.loginfo("RealSense pipeline stopped successfully.")
            except RuntimeError as e:
                rospy.logerr(f"Error stopping RealSense pipeline during shutdown: {e}")
        
        if self.show_cv_window:
            rospy.loginfo("Destroying OpenCV windows.")
            cv2.destroyAllWindows()
        rospy.loginfo(f"Node '{rospy.get_name()}' shutdown complete.")

if __name__ == '__main__':
    try:
        detector = ArucoRealSenseDetector() # 创建检测器对象 (会调用 __init__)
        # 检查在 __init__ 过程中 rospy 是否已经被要求关闭 (例如，如果 RealSense 启动失败)
        if not rospy.is_shutdown():
            detector.run() # 进入主循环
    except rospy.ROSInterruptException:
        rospy.loginfo(f"ROS node '{rospy.get_name()}' interrupted by Ctrl+C or ROS master shutdown.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in '{rospy.get_name()}': {e}")
        import traceback
        rospy.logerr(traceback.format_exc()) # 打印完整的 Python 异常堆栈
    finally:
        # 这是一个最终的保障措施，确保在任何情况下（即使有未捕获的异常导致run未完成）
        # 如果窗口被显示了，都尝试关闭它。
        # 但更好的做法是在 shutdown_hook 中处理。
        # if rospy.get_param("~show_cv_window", True) and not rospy.is_shutdown():
        #    cv2.destroyAllWindows()
        rospy.loginfo(f"'{rospy.get_name()}' script finished.")