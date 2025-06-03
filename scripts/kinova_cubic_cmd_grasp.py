#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf.transformations as tf_trans # Still useful for euler operations if needed

# --- 导入自定义消息 ---
from ee368_project.msg import PickAndPlaceGoalInCamera # Assuming this is your custom message
from std_msgs.msg import String

# --- 导入 IK 和 Trajectory Executor ---
try:
    import inverse_kinematics as ik
    from kinova_cubic_trajectory_executor import KinovaCubicTrajectoryExecutor
except ImportError as e:
    rospy.logerr(f"Failed to import custom IK or Trajectory Executor: {e}")
    rospy.logerr("Make sure inverse_kinematics.py and kinova_cubic_trajectory_executor.py are in the same directory or Python path.")
    import sys
    sys.exit(1)

# --- 导入 Gripper Controller (SimplifiedArmController from move_cartesian.py) ---
# We still need this for gripper commands as KinovaCubicTrajectoryExecutor only handles arm motion.
try:
    from move_cartesian import SimplifiedArmController # Assuming this is where it lives
except ImportError as e:
    rospy.logwarn(f"Failed to import SimplifiedArmController for gripper: {e}. Gripper commands will not be available.")
    SimplifiedArmController = None


class KinovaCubicCmdGrasp:
    # --- 机械臂状态常量 ---
    STATE_IDLE = 0
    STATE_BUSY = 1
    STATE_ERROR = 2

    # --- 机械臂状态对应的发布消息 ---
    MSG_ARM_IDLE = "0"
    MSG_ARM_BUSY = "1"
    MSG_ARM_ERROR = "2"

    def __init__(self):
        rospy.init_node('kinova_cubic_cmd_grasp_controller', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite") # Used by SimplifiedArmController
        rospy.loginfo(f"KinovaCubicCmdGrasp 使用机器人名称: {self.robot_name}")

        # --- 机械臂状态发布器 ---
        self.arm_status_pub = rospy.Publisher(
            f"/{self.robot_name}/arm_status", String, queue_size=1, latch=True
        )
        self.current_arm_state = None # 初始化当前机械臂内部状态

        # --- Trajectory Executor for Arm Motion ---
        # KinovaCubicTrajectoryExecutor handles its own ROS node initialization internally if not already done
        # but it's better to have the main node init first.
        # It also handles joint state subscription and DH parameters via inverse_kinematics.py
        self.trajectory_executor = KinovaCubicTrajectoryExecutor() # This will init its own ROS components

        # --- SimplifiedArmController for Gripper ---
        self.gripper_controller = None
        if SimplifiedArmController:
            self.gripper_controller = SimplifiedArmController(robot_name_param=self.robot_name)
            if not self.gripper_controller.is_init_success:
                rospy.logwarn("Gripper controller (SimplifiedArmController) failed to initialize. Gripper will not function.")
                self.gripper_controller = None # Explicitly set to None if init failed
        else:
            rospy.logwarn("SimplifiedArmController not available. Gripper will not function.")

        self.is_arm_motion_ready = False # Based on trajectory_executor
        self.is_gripper_ready = bool(self.gripper_controller and self.gripper_controller.is_gripper_present)


        # Wait for trajectory_executor to be ready (connected to action server and receiving joint states)
        rospy.loginfo("Waiting for Trajectory Executor to be ready...")
        initial_angles_test = self.trajectory_executor.get_current_joint_angles(timeout_sec=10.0)
        if initial_angles_test is not None:
            rospy.loginfo("Trajectory Executor is ready and receiving joint states.")
            self.is_arm_motion_ready = True
        else:
            rospy.logerr("Trajectory Executor failed to get initial joint states. Arm motion will not function.")
            self._update_and_publish_arm_status(self.STATE_ERROR)
            # Note: trajectory_executor might have already shut down if action server not found.
            # We might want to exit here if critical components are missing.
            if not self.trajectory_executor.trajectory_client.gh: # Check if client is valid
                 rospy.logerr("Trajectory action client not available. Shutting down.")
                 rospy.signal_shutdown("Critical component TrajectoryExecutor failed.")
                 sys.exit(1)


        # --- DH Parameters and Num Joints (from ik module) ---
        self.dh_params = ik.dh_parameters
        self.num_joints = ik.num_joints

        # --- Transformations ---
        # End-effector to Camera transform (meters for translation)
        T_ee_camera_mm_translation = np.array(rospy.get_param('~T_ee_camera_translation_mm', [60.0, -40.0, -110.0]))
        # Using a more standard ZYX Euler for R_ee_camera, or provide full matrix
        # Defaulting to your R_ee_camera: [[0., -1.,  0.], [1.,  0.,  0.], [0.,  0.,  1.]]
        R_ee_camera_matrix_param = rospy.get_param('~R_ee_camera_matrix', [[0., -1.,  0.], [1.,  0.,  0.], [0.,  0.,  1.]])
        R_ee_camera = np.array(R_ee_camera_matrix_param)

        self.T_ee_camera = np.identity(4)
        self.T_ee_camera[:3,:3] = R_ee_camera
        self.T_ee_camera[0:3, 3] = T_ee_camera_mm_translation / 1000.0
        rospy.loginfo(f"Using T_ee_camera (End-Effector to Camera Transform):\n{self.T_ee_camera}")

        # --- Gripper Orientation (Fixed for Pick/Place) ---
        # Euler angles [roll, pitch, yaw] in degrees, relative to robot base
        self.fixed_gripper_orientation_base_deg = np.array(rospy.get_param(
            '~fixed_gripper_orientation_base_deg', [0.0, 180.0, 0.0]
        ))
        self.fixed_gripper_orientation_base_rad = np.deg2rad(self.fixed_gripper_orientation_base_deg)
        rospy.loginfo(f"Using fixed gripper orientation (Euler XYZ rad, base frame): {self.fixed_gripper_orientation_base_rad}")

        # --- Z Offsets ---
        self.pick_z_offset_meters = rospy.get_param("~pick_z_offset_meters", 0.005) # Small positive to grasp above surface
        rospy.loginfo(f"Using pick Z offset (from detected pick point): {self.pick_z_offset_meters} m")
        
        self.place_z_offset_meters = rospy.get_param("~place_z_offset_meters", 0.01) # Small positive to place above surface
        rospy.loginfo(f"Using place Z offset (from detected place point): {self.place_z_offset_meters} m")

        self.pre_action_z_lift_meters = rospy.get_param("~pre_action_z_lift_meters", 0.05)
        rospy.loginfo(f"Using pre-action Z lift height: {self.pre_action_z_lift_meters} m")

        # --- Home Pose Definition (Cartesian) ---
        # Define a "home" in joint space, then find its Cartesian equivalent
        q_home_joints_rad = np.array(rospy.get_param('~home_joint_angles_deg', [30.66, 346.57, 72.23, 270.08, 265.45, 345.69])) * np.pi / 180.0
        if len(q_home_joints_rad) != self.num_joints:
            rospy.logwarn(f"Provided home_joint_angles_deg length mismatch. Using all zeros.")
            q_home_joints_rad = np.zeros(self.num_joints)
        
        T_home_cartesian, _ = ik.forward_kinematics(q_home_joints_rad, self.dh_params)
        self.home_cartesian_position = T_home_cartesian[:3,3]
        self.home_cartesian_orientation_euler = R.from_matrix(T_home_cartesian[:3,:3]).as_euler('xyz', degrees=False)
        rospy.loginfo(f"Home pose (Cartesian) derived from joints {np.round(np.rad2deg(q_home_joints_rad),1)} deg:")
        rospy.loginfo(f"  Position: {np.round(self.home_cartesian_position,3)} m")
        rospy.loginfo(f"  Orientation (Euler XYZ rad): {np.round(self.home_cartesian_orientation_euler,3)}")


        # --- Subscriber for Pick and Place Goals ---
        self.pick_and_place_sub = rospy.Subscriber(
            "kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera,
            self.pick_and_place_callback,
            queue_size=1
        )
        rospy.loginfo("Kinova Cubic Command Grasp Controller initialized. Waiting for goals...")

        # --- Initial Homing and State Setting ---
        if self.is_arm_motion_ready: # Gripper readiness is not critical for homing the arm
            rospy.loginfo("Performing initial homing sequence...")
            if not self._go_to_home_safe_position():
                 rospy.logwarn("Initial homing failed. Arm may not be in a known safe state.")
                 # Status will be set by _go_to_home_safe_position (BUSY then IDLE or ERROR)
            # _go_to_home_safe_position will set status to IDLE if successful
        else:
            rospy.logerr("Arm motion system not ready. Cannot home. Setting state to ERROR.")
            self._update_and_publish_arm_status(self.STATE_ERROR)


    def _update_and_publish_arm_status(self, new_state):
        if new_state != self.current_arm_state:
            self.current_arm_state = new_state
            status_msg_str = ""
            if new_state == self.STATE_IDLE: status_msg_str = self.MSG_ARM_IDLE
            elif new_state == self.STATE_BUSY: status_msg_str = self.MSG_ARM_BUSY
            elif new_state == self.STATE_ERROR: status_msg_str = self.MSG_ARM_ERROR
            else: status_msg_str = f"Unknown State ({new_state})"
            
            rospy.loginfo(f"Arm status changed to: {status_msg_str} (Internal state: {new_state})")
            self.arm_status_pub.publish(String(data=status_msg_str))

    def get_current_T_base_camera(self):
        current_q_rad = self.trajectory_executor.get_current_joint_angles()
        if current_q_rad is None:
            rospy.logerr("Could not get current joint angles from trajectory executor.")
            return None
        
        T_base_to_ee, _ = ik.forward_kinematics(current_q_rad, self.dh_params)
        # FK from ik.py might return T_base_to_ee with translation in mm or m.
        # The example ik.py returns in meters implicitly by DH table (d units). Check your DH.
        # If DH 'd' units are in mm, you'd need: T_base_to_ee[0:3, 3] /= 1000.0
        # Assuming DH parameters are in meters for lengths.

        T_base_camera = T_base_to_ee @ self.T_ee_camera
        return T_base_camera

    def _transform_point_camera_to_base(self, point_in_camera_msg, T_base_camera_current):
        P_camera_homogeneous = np.array([point_in_camera_msg.x,
                                         point_in_camera_msg.y,
                                         point_in_camera_msg.z,
                                         1.0]).reshape(4, 1)
        P_base_homogeneous = T_base_camera_current @ P_camera_homogeneous
        return P_base_homogeneous[0:3, 0].flatten() # Return as 1D array

    def _go_to_home_safe_position(self, duration=8.0):
        rospy.loginfo("Attempting to move to home/safe Cartesian position...")
        self._update_and_publish_arm_status(self.STATE_BUSY)

        success = self.trajectory_executor.plan_and_execute_to_pose(
            target_position=self.home_cartesian_position,
            target_orientation_euler=self.home_cartesian_orientation_euler,
            trajectory_duration=duration
        )
        if success:
            rospy.loginfo("Successfully moved to home/safe position.")
            self._update_and_publish_arm_status(self.STATE_IDLE)
            return True
        else:
            rospy.logwarn("Moving to home/safe position failed.")
            # Consider if error state is more appropriate if home is critical
            # For now, let's assume it might recover, so keep it BUSY or try to re-idle.
            # If it failed, it's likely not IDLE. An ERROR state might be better if unrecoverable.
            self._update_and_publish_arm_status(self.STATE_ERROR) # If homing fails, system is in uncertain state.
            return False

    def _perform_gripper_action(self, open_percentage):
        """
        Commands the gripper. open_percentage: 0 (closed) to 1 (fully open).
        Specific mapping to your 0-100% might be needed.
        The example SimplifiedArmController uses 0 for fully closed, 1 for fully open.
        The original grasp.py used 0.7 for "open for grasp", 0.77 for "grasp closure".
        """
        if not self.is_gripper_ready:
            rospy.logwarn("Gripper not ready or not present. Skipping gripper action.")
            return False # Or True if skipping is acceptable. Let's say False to indicate an issue.

        rospy.loginfo(f"Commanding gripper to {open_percentage*100:.0f}%...")
        # The gripper controller might need specific values (e.g. 0-1, or 0-100)
        # Adjust if your move_gripper expects something else
        success = self.gripper_controller.move_gripper(open_percentage) 
        if not success:
            rospy.logwarn("Gripper command failed.")
        rospy.sleep(1.5) # Allow time for gripper to act
        return success

    def _perform_action_sequence(self, action_name, target_xyz_base, z_offset_specific, is_pick_action, move_duration=5.0):
        rospy.loginfo(f"--- Starting {action_name} sequence ---")
        
        # Target pose for the actual action (pick/place)
        action_target_position = np.copy(target_xyz_base)
        action_target_position[2] += z_offset_specific # Apply Z offset for the action itself
        action_target_orientation_euler = self.fixed_gripper_orientation_base_rad

        # Pre-action pose (lifted above the action point)
        pre_action_position = np.copy(action_target_position)
        pre_action_position[2] += self.pre_action_z_lift_meters
        pre_action_orientation_euler = action_target_orientation_euler # Same orientation

        # 1. Move to Pre-Action Position (above target)
        rospy.loginfo(f"Step 1 ({action_name}): Moving to pre-{action_name.lower()} position (X:{pre_action_position[0]:.3f}, Y:{pre_action_position[1]:.3f}, Z:{pre_action_position[2]:.3f})...")
        if not self.trajectory_executor.plan_and_execute_to_pose(
            pre_action_position, pre_action_orientation_euler, trajectory_duration=move_duration):
            rospy.logwarn(f"Failed to move to pre-{action_name.lower()} position. Aborting {action_name}.")
            return False

        # 2. Prepare Gripper (if pick) or Approach (if place)
        if is_pick_action:
            rospy.loginfo(f"Step 2 ({action_name}): Opening gripper for grasp...")
            open_val = rospy.get_param("~gripper_open_for_pick_percentage", 0.7) # e.g. 70% open
            if not self._perform_gripper_action(open_val):
                rospy.logwarn("Failed to open gripper for pick. Continuing cautiously...")
        else: # is_place_action
            rospy.loginfo(f"Step 2 ({action_name}): Gripper already holding object (presumably). Ready to approach place Z.")
            pass # Gripper state should be "closed" from pick

        # 3. Move to Exact Action Z Position
        rospy.loginfo(f"Step 3 ({action_name}): Moving to {action_name.lower()} Z position (X:{action_target_position[0]:.3f}, Y:{action_target_position[1]:.3f}, Z:{action_target_position[2]:.3f})...")
        if not self.trajectory_executor.plan_and_execute_to_pose(
            action_target_position, action_target_orientation_euler, trajectory_duration=max(2.0, move_duration/2)): # Slower for precision
            rospy.logwarn(f"Failed to move to {action_name.lower()} Z position. Aborting {action_name}.")
            # Attempt to lift back to pre-action if possible, then fail
            self.trajectory_executor.plan_and_execute_to_pose(pre_action_position, pre_action_orientation_euler, trajectory_duration=3.0)
            return False
        rospy.sleep(0.5) # Settle

        # 4. Actuate Gripper
        if self.is_gripper_ready:
            if is_pick_action:
                rospy.loginfo(f"Step 4 ({action_name}): Closing gripper to grasp...")
                grasp_val = rospy.get_param("~gripper_grasp_closure_percentage", 0.1) # e.g. 10% (quite closed)
                if not self._perform_gripper_action(grasp_val):
                    rospy.logwarn(f"Failed to close gripper for {action_name}. Object may not be grasped.")
                    # Decide if this is a fatal error for the sequence
            else: # is_place_action
                rospy.loginfo(f"Step 4 ({action_name}): Opening gripper to release...")
                release_val = rospy.get_param("~gripper_release_percentage", 0.7) # e.g. 70% open
                if not self._perform_gripper_action(release_val):
                     rospy.logwarn(f"Failed to open gripper for {action_name}. Object may not be released.")
        else:
            rospy.logwarn("Gripper not available, skipping Step 4 action.")


        # 5. Lift/Retreat from Action Point (back to Pre-Action Position)
        rospy.loginfo(f"Step 5 ({action_name}): Lifting/Retreating from {action_name.lower()} point...")
        if not self.trajectory_executor.plan_and_execute_to_pose(
            pre_action_position, pre_action_orientation_euler, trajectory_duration=move_duration):
            rospy.logwarn(f"Failed to lift/retreat from {action_name.lower()} point. Arm may be in an awkward position.")
            # This is a potentially problematic failure state
            return False # Indicate overall sequence failure

        rospy.loginfo(f"--- {action_name} sequence completed ---")
        return True


    def pick_and_place_callback(self, msg: PickAndPlaceGoalInCamera):
        if not self.is_arm_motion_ready:
            rospy.logerr("Arm motion system not ready. Ignoring pick and place goal.")
            self._update_and_publish_arm_status(self.STATE_ERROR)
            return

        if self.current_arm_state == self.STATE_BUSY:
            rospy.logwarn(f"Arm is BUSY. Ignoring new pick and place goal for '{msg.object_id_at_pick}'.")
            return
        
        self._update_and_publish_arm_status(self.STATE_BUSY)

        rospy.loginfo(f"Received P&P Goal: Pick '{msg.object_id_at_pick}' (cam_frame), Place at '{msg.target_location_id_at_place}' (cam_frame).")
        
        T_base_camera_snapshot = self.get_current_T_base_camera()
        if T_base_camera_snapshot is None:
            rospy.logerr("Failed to get T_base_camera at start of operation. Aborting.")
            self._go_to_home_safe_position() # Attempt to return to safe state
            return # Status already set by _go_to_home_safe_position

        rospy.loginfo(f"  Using T_base_camera snapshot for this operation:\n{np.round(T_base_camera_snapshot, 3)}")
        
        pick_point_base_xyz = self._transform_point_camera_to_base(msg.pick_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Pick Point (base frame): {np.round(pick_point_base_xyz,3)}")
        
        place_point_base_xyz = self._transform_point_camera_to_base(msg.place_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Place Point (base frame): {np.round(place_point_base_xyz,3)}")

        # --- Perform Pick ---
        pick_successful = self._perform_action_sequence(
            action_name="Pick",
            target_xyz_base=pick_point_base_xyz,
            z_offset_specific=self.pick_z_offset_meters,
            is_pick_action=True,
            move_duration=rospy.get_param("~pick_move_duration", 6.0)
        )

        if not pick_successful:
            rospy.logwarn("Pick sequence failed. Aborting P&P operation.")
            self._go_to_home_safe_position()
            return

        # --- Perform Place ---
        place_successful = self._perform_action_sequence(
            action_name="Place",
            target_xyz_base=place_point_base_xyz,
            z_offset_specific=self.place_z_offset_meters,
            is_pick_action=False, # This is a place action
            move_duration=rospy.get_param("~place_move_duration", 7.0)
        )

        if not place_successful:
            rospy.logwarn("Place sequence failed.")
            # Object might be dropped, or still held if gripper didn't open.
            # Or arm failed to move.
        
        rospy.loginfo("Pick and Place attempt finished. Returning to home position.")
        self._go_to_home_safe_position() # This will set state to IDLE or ERROR

    def run(self):
        if not self.is_arm_motion_ready:
            rospy.logerr(f"{rospy.get_name()} cannot run effectively as arm motion system is not ready.")
            # Status should already be ERROR from __init__
            # Loop briefly to allow ROS to publish the error state if not shutting down.
            rate = rospy.Rate(1)
            for _ in range(3):
                if rospy.is_shutdown(): break
                rate.sleep()
            return

        rospy.loginfo(f"{rospy.get_name()} is running and awaiting P&P goals...")
        # Initial status is set during _go_to_home_safe_position or error in init
        rospy.spin()
        rospy.loginfo(f"{rospy.get_name()} shutting down.")

if __name__ == '__main__':
    try:
        controller_node = KinovaCubicCmdGrasp()
        controller_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Kinova Cubic Command Grasp Controller interrupted.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred in KinovaCubicCmdGrasp: {e}")
        import traceback
        traceback.print_exc()