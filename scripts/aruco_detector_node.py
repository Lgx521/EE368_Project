#!/usr/bin/env python3

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
# Import your custom messages
from ee368_project.msg import ArucoMarker, ArucoMarkerArray
import tf.transformations as tf_trans # For converting rotation matrix to quaternion

class ArucoDetectorROS:
    def __init__(self):
        rospy.init_node('aruco_detector_node', anonymous=False)

        # --- Parameters ---
        self.marker_real_size_meters = rospy.get_param("~marker_size", 0.05) # Default 5cm
        aruco_dict_name_param = rospy.get_param("~aruco_dictionary_name", "DICT_6X6_250")
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_color_optical_frame")
        self.show_cv_window = rospy.get_param("~show_cv_window", False) # For debugging

        # Resolve Aruco dictionary name to OpenCV constant
        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None:
                raise AttributeError
        except AttributeError:
            rospy.logerr(f"Invalid ArUco dictionary name: {aruco_dict_name_param}. Using DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        # Compatibility for DetectorParameters
        try:
            self.parameters = aruco.DetectorParameters() # OpenCV 4.7+
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create() # Older OpenCV

        # --- RealSense Initialization ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = rospy.get_param("~color_width", 640)
        self.color_height = rospy.get_param("~color_height", 480)
        self.color_fps = rospy.get_param("~color_fps", 30)

        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            rospy.logerr(f"Failed to start RealSense pipeline: {e}")
            rospy.signal_shutdown("RealSense pipeline failed to start.")
            return

        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [self.intrinsics_color.fx, 0, self.intrinsics_color.ppx],
            [0, self.intrinsics_color.fy, self.intrinsics_color.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array(self.intrinsics_color.coeffs, dtype=np.float32)
        if self.dist_coeffs is None or len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        elif not (len(self.dist_coeffs) in [4, 5, 8, 12, 14]):
            rospy.logwarn(f"Unexpected number of distortion coefficients ({len(self.dist_coeffs)}). Using zero distortion.")
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)

        rospy.loginfo("RealSense Camera Intrinsics (fx, fy, cx, cy):")
        rospy.loginfo(f"  fx: {self.intrinsics_color.fx:.2f}, fy: {self.intrinsics_color.fy:.2f}")
        rospy.loginfo(f"  cx: {self.intrinsics_color.ppx:.2f}, cy: {self.intrinsics_color.ppy:.2f}")
        rospy.loginfo(f"Distortion Coefficients: {self.dist_coeffs.flatten()}")

        # --- ROS Publishers and Bridge ---
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("aruco_detector/image_processed", Image, queue_size=1)
        self.marker_pub = rospy.Publisher("aruco_detector/markers", ArucoMarkerArray, queue_size=1)

        rospy.loginfo("ArUco detector node initialized.")
        rospy.on_shutdown(self.shutdown_hook)


    def rotation_matrix_to_quaternion(self, R):
        """Convert a rotation matrix to a quaternion."""
        q = tf_trans.quaternion_from_matrix(np.vstack((np.hstack((R, np.zeros((3,1)))), [0,0,0,1])))
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000) # Timeout to prevent indefinite block
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"Timeout waiting for frames: {e}")
            return

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            rospy.logwarn_throttle(5, "No color frame received")
            return

        color_image = np.asanyarray(color_frame_rs.get_data())
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        current_time = rospy.Time.now()
        display_image = color_image.copy()
        marker_array_msg = ArucoMarkerArray()
        marker_array_msg.header.stamp = current_time
        marker_array_msg.header.frame_id = self.camera_frame_id 

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            # Estimate pose for each marker
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Draw axis (compatibility for older OpenCV)
                try:
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                except AttributeError:
                    aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                
                # Create ArucoMarker message
                marker_msg = ArucoMarker()
                marker_msg.header.stamp = current_time
                marker_msg.header.frame_id = self.camera_frame_id 
                marker_msg.id = int(marker_id) # Ensure it's a standard int

                # Position
                marker_msg.pose.position.x = tvec[0][0]
                marker_msg.pose.position.y = tvec[0][1]
                marker_msg.pose.position.z = tvec[0][2]

                # Orientation (convert rvec to quaternion)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                marker_msg.pose.orientation = self.rotation_matrix_to_quaternion(rotation_matrix)
                
                marker_array_msg.markers.append(marker_msg)
        
        # Publish detected markers
        if marker_array_msg.markers:
            self.marker_pub.publish(marker_array_msg)

        # Publish processed image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            img_msg.header.stamp = current_time
            img_msg.header.frame_id = self.camera_frame_id # Or a more generic frame if needed
            self.image_pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        if self.show_cv_window:
            cv2.imshow("ArUco Detection (ROS)", display_image)
            cv2.waitKey(1)


    def run(self):
        rate = rospy.Rate(self.color_fps) # Match camera FPS or a bit lower
        while not rospy.is_shutdown():
            self.process_frame()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS Interrupt. Shutting down.")
                break
        
        if self.show_cv_window:
            cv2.destroyAllWindows()


    def shutdown_hook(self):
        rospy.loginfo("Stopping RealSense pipeline...")
        if hasattr(self, 'pipeline'): # Check if pipeline was initialized
            self.pipeline.stop()
        rospy.loginfo("Shutdown complete.")


if __name__ == '__main__':
    try:
        detector = ArucoDetectorROS()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Aruco detector node interrupted.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception in ArUco detector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rospy.get_param("~show_cv_window", False): # ensure cv window is closed if used
             cv2.destroyAllWindows()