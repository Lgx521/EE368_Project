#!/usr/bin/env python3

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from ee368_project.msg import ArucoMarker, ArucoMarkerArray # 使用你的包名
import tf.transformations as tf_trans
import message_filters # For synchronizing image and camera_info

class ArucoSubscriberDetector:
    def __init__(self):
        rospy.init_node('aruco_subscriber_node', anonymous=False)

        # --- Parameters ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # Default 5cm
        aruco_dict_name_param = rospy.get_param("~aruco_dictionary_name", "DICT_6X6_250")
        self.show_cv_window = rospy.get_param("~show_cv_window", False) # For debugging
        
        # Topics to subscribe to (from realsense2_camera node)
        # 确保这些话题名称与 realsense2_camera 节点发布的实际话题名称一致！
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        # Resolve Aruco dictionary name
        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None: raise AttributeError
        except AttributeError:
            rospy.logerr(f"Invalid ArUco dictionary name: {aruco_dict_name_param}. Using DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250
        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        try:
            self.parameters = aruco.DetectorParameters() # OpenCV 4.7+
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create() # Older OpenCV

        # --- ROS Subscribers, Publishers, and Bridge ---
        self.bridge = CvBridge()
        
        # Subscribers using ApproximateTimeSynchronizer
        # 等待图像和相机信息都到达后再处理
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.camera_info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        
        # TimeSynchronizer to get synced image and camera_info messages
        # queue_size 建议设为10左右，slop 设为0.1秒 (100ms)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.camera_info_sub], 
            queue_size=10, 
            slop=0.1, # Allow 0.1 seconds difference
            allow_headerless=False # CameraInfo must have a header
        )
        self.ts.registerCallback(self.image_camera_info_callback)

        # Publishers
        self.processed_image_pub = rospy.Publisher("aruco_detector/image_processed", Image, queue_size=1)
        self.marker_array_pub = rospy.Publisher("aruco_detector/markers", ArucoMarkerArray, queue_size=1)
        # 我们不再自己发布 CameraInfo，因为我们是从 realsense2_camera 订阅的
        # 如果 RViz 的 Camera Display 需要，它可以直接订阅 realsense2_camera 的 CameraInfo 话题

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame_id = None # Will be set from CameraInfo message

        rospy.loginfo(f"ArUco Subscriber Node initialized. Subscribing to image on '{self.image_topic}' and info on '{self.camera_info_topic}'.")
        rospy.loginfo(f"Publishing processed image to 'aruco_detector/image_processed'")
        rospy.loginfo(f"Publishing markers to 'aruco_detector/markers'")

    def image_camera_info_callback(self, image_msg, camera_info_msg):
        if self.camera_matrix is None: # Populate camera intrinsics once
            self.camera_matrix = np.array(camera_info_msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info_msg.D)
            self.camera_frame_id = camera_info_msg.header.frame_id
            rospy.loginfo(f"Received CameraInfo. Frame ID: {self.camera_frame_id}")
            rospy.loginfo(f"K: \n{self.camera_matrix}")
            rospy.loginfo(f"D: {self.dist_coeffs}")

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        display_image = cv_image.copy() # For drawing
        marker_array_msg = ArucoMarkerArray()
        # 使用 camera_info_msg 的 header，因为它与图像是同步的
        marker_array_msg.header = camera_info_msg.header 
        # frame_id 应该与 camera_info_msg.header.frame_id 相同

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i]

                    try: # Draw axis (compatibility)
                        cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                    except AttributeError:
                        aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                    
                    marker_msg = ArucoMarker()
                    marker_msg.header = camera_info_msg.header # Use synced header
                    marker_msg.id = int(marker_id)
                    marker_msg.pose.position = Point(x=tvec[0,0], y=tvec[0,1], z=tvec[0,2])
                    
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    q = tf_trans.quaternion_from_matrix(np.vstack((np.hstack((rotation_matrix, tvec.reshape(3,1))), [[0,0,0,1]])))
                    marker_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                    
                    marker_array_msg.markers.append(marker_msg)
            else:
                rospy.logwarn_throttle(5.0, "Camera intrinsics not yet available for pose estimation.")
        
        # Publish detected markers
        if marker_array_msg.markers:
            self.marker_array_pub.publish(marker_array_msg)

        # Publish processed image
        try:
            processed_img_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            processed_img_msg.header = image_msg.header # Preserve original image header (timestamp, frame_id)
            self.processed_image_pub.publish(processed_img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error publishing processed image: {e}")

        if self.show_cv_window:
            cv2.imshow("ArUco Detection (Subscribed)", display_image)
            cv2.waitKey(1)

    def run(self):
        rospy.spin() # Keeps python from exiting until this node is stopped
        if self.show_cv_window:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = ArucoSubscriberDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Aruco subscriber node interrupted.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception in ArUco subscriber: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rospy.get_param("~show_cv_window", False): # ensure cv window is closed if used
             cv2.destroyAllWindows()