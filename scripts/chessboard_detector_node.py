#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import Point
from std_msgs.msg import Header # For the header in the custom message
# Import your custom message
from ee368_project.msg import ChessboardCorners # Make sure package name is correct

class ChessboardDetectorROS:
    def __init__(self):
        rospy.init_node('chessboard_detector_node', anonymous=False)

        # --- Parameters for Chessboard ---
        # Number of inner corners per a chessboard row and column
        self.pattern_width = rospy.get_param("~chessboard_pattern_width", 9) # e.g., for a 10x7 board, this is 9
        self.pattern_height = rospy.get_param("~chessboard_pattern_height", 6) # e.g., for a 10x7 board, this is 6
        self.chessboard_pattern_size = (self.pattern_width, self.pattern_height)

        # --- General Parameters ---
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_color_optical_frame")
        self.show_cv_window = rospy.get_param("~show_cv_window", True) # Default to True for easier debugging

        # --- RealSense Initialization ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = rospy.get_param("~color_width", 1280)
        self.color_height = rospy.get_param("~color_height", 720)
        self.depth_width = rospy.get_param("~depth_width", 1280) # Match color for easier alignment
        self.depth_height = rospy.get_param("~depth_height", 720)
        self.fps = rospy.get_param("~fps", 30)

        self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)

        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            rospy.logerr(f"Failed to start RealSense pipeline: {e}")
            rospy.signal_shutdown("RealSense pipeline failed to start.")
            return

        # Alignment
        self.align_to_color = rs.align(rs.stream.color)

        # Intrinsics
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Depth scale (for converting depth frame raw values to meters, if needed, though get_distance() is preferred)
        # depth_sensor = self.profile.get_device().first_depth_sensor()
        # self.depth_scale = depth_sensor.get_depth_scale()

        rospy.loginfo("RealSense Camera Intrinsics (fx, fy, cx, cy):")
        rospy.loginfo(f"  fx: {self.intrinsics_color.fx:.2f}, fy: {self.intrinsics_color.fy:.2f}")
        rospy.loginfo(f"  cx: {self.intrinsics_color.ppx:.2f}, cy: {self.intrinsics_color.ppy:.2f}")

        # --- ROS Publisher ---
        self.corners_pub = rospy.Publisher("/chessboard_corners", ChessboardCorners, queue_size=10)

        # OpenCV criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        rospy.loginfo("Chessboard detector node initialized. Publishing to /chessboard_corners")
        rospy.on_shutdown(self.shutdown_hook)

    def get_3d_point(self, depth_frame, u, v):
        """
        Get 3D coordinates (X, Y, Z) for a given pixel (u, v) using depth data.
        Returns a geometry_msgs/Point or None if depth is invalid.
        """
        try:
            depth_value_meters = depth_frame.get_distance(int(u), int(v))
            if depth_value_meters == 0: # Invalid depth
                rospy.logwarn_throttle(5, f"Invalid depth (0) at pixel ({int(u)}, {int(v)})")
                return None

            # Deproject pixel to 3D point in camera coordinates
            # rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_in_meters)
            point_3d_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics_color, [float(u), float(v)], depth_value_meters)
            
            ros_point = Point()
            ros_point.x = point_3d_cam[0]
            ros_point.y = point_3d_cam[1]
            ros_point.z = point_3d_cam[2]
            return ros_point
            
        except Exception as e:
            rospy.logerr_throttle(5, f"Error deprojecting pixel ({u},{v}): {e}")
            return None


    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000) # Increased timeout
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"Timeout waiting for frames: {e}")
            return

        # Align the depth frame to color frame
        aligned_frames = self.align_to_color.process(frames)
        
        color_frame_rs = aligned_frames.get_color_frame()
        depth_frame_rs = aligned_frames.get_depth_frame()

        if not color_frame_rs or not depth_frame_rs:
            rospy.logwarn_throttle(5, "No color or depth frame received")
            return

        color_image = np.asanyarray(color_frame_rs.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        display_image = color_image.copy() # For drawing

        # Find chessboard corners
        ret, corners_2d_raw = cv2.findChessboardCorners(gray_image, self.chessboard_pattern_size, None)

        if ret:
            # Refine corner locations
            corners_2d_refined = cv2.cornerSubPix(gray_image, corners_2d_raw, (11, 11), (-1, -1), self.criteria)
            
            # Draw corners on the display image
            cv2.drawChessboardCorners(display_image, self.chessboard_pattern_size, corners_2d_refined, ret)

            # Extract the four outer corners (pixel coordinates)
            # corners_2d_refined is a (Nx1x2) array, where N = pattern_width * pattern_height
            # Order: top-left, moving across rows, then down columns
            
            # Top-left inner corner of the pattern
            top_left_inner_px = corners_2d_refined[0].ravel() 
            # Top-right inner corner of the pattern
            top_right_inner_px = corners_2d_refined[self.pattern_width - 1].ravel()
            # Bottom-left inner corner of the pattern
            bottom_left_inner_px = corners_2d_refined[(self.pattern_height - 1) * self.pattern_width].ravel()
            # Bottom-right inner corner of the pattern
            bottom_right_inner_px = corners_2d_refined[self.pattern_height * self.pattern_width - 1].ravel()

            # Get 3D coordinates for these outer corners
            point_tl = self.get_3d_point(depth_frame_rs, top_left_inner_px[0], top_left_inner_px[1])
            point_tr = self.get_3d_point(depth_frame_rs, top_right_inner_px[0], top_right_inner_px[1])
            point_bl = self.get_3d_point(depth_frame_rs, bottom_left_inner_px[0], bottom_left_inner_px[1])
            point_br = self.get_3d_point(depth_frame_rs, bottom_right_inner_px[0], bottom_right_inner_px[1])

            # Only publish if all four points are valid
            if all([point_tl, point_tr, point_bl, point_br]):
                corners_msg = ChessboardCorners()
                corners_msg.header.stamp = rospy.Time.now()
                corners_msg.header.frame_id = self.camera_frame_id
                
                corners_msg.top_left = point_tl
                corners_msg.top_right = point_tr
                corners_msg.bottom_left = point_bl
                corners_msg.bottom_right = point_br
                
                self.corners_pub.publish(corners_msg)
                rospy.loginfo_once("Chessboard corners published.") # Log once to reduce console spam
            else:
                rospy.logwarn_throttle(5, "Could not get 3D coordinates for all four chessboard corners.")
        else:
            rospy.logwarn_throttle(10, "Chessboard not detected.")


        if self.show_cv_window:
            cv2.imshow("Chessboard Detection", display_image)
            key = cv2.waitKey(1)
            if key == 27: # ESC key
                rospy.signal_shutdown("ESC key pressed")
                cv2.destroyAllWindows()


    def run(self):
        rate = rospy.Rate(self.fps)
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
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if self.show_cv_window:
            cv2.destroyAllWindows()
        rospy.loginfo("Shutdown complete.")


if __name__ == '__main__':
    try:
        detector = ChessboardDetectorROS()
        if hasattr(detector, 'profile'): # Check if pipeline started successfully
            detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Chessboard detector node interrupted.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception in Chessboard detector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure OpenCV windows are closed on any exit if they were shown
        if rospy.is_shutdown() and rospy.get_param("~show_cv_window", False):
             cv2.destroyAllWindows()