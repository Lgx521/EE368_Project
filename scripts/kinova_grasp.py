#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

# --- 导入自定义消息 ---
from ee368_project.msg import PickAndPlaceGoalInCamera

# --- 从同一包的scripts目录导入其他模块 ---
try:
    from move_cartesian import SimplifiedArmController
    from forward_kinematics import arm as FKCalculator
except ImportError as e:
    rospy.logerr(f"Failed to import SimplifiedArmController or FKCalculator: {e}")
    import sys
    sys.exit(1)

from sensor_msgs.msg import JointState
from kortex_driver.msg import BaseCyclic_Feedback

class KinovaPickAndPlaceController:
    def __init__(self):
        rospy.init_node('kinova_pick_and_place_controller', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")
        rospy.loginfo(f"KinovaPickAndPlaceController using robot_name: {self.robot_name}")

        self.arm_controller = SimplifiedArmController(robot_name_param=self.robot_name)
        self.is_arm_ready = False

        if not self.arm_controller.is_init_success:
            rospy.logerr("Arm controller initialization failed.")
            return
        if self.arm_controller.clear_robot_faults() and self.arm_controller.activate_notifications():
            self.is_arm_ready = True
            rospy.loginfo("Arm is ready and notifications activated.")
        else:
            rospy.logerr("Failed to initialize arm.")
            return

        self.fk_calculator = FKCalculator(Dof=6)

        T_ee_camera_mm_translation = np.array([60.0, 0.0, -110.0])
        R_ee_camera = np.array([[0., -1.,  0.], [1.,  0.,  0.], [0.,  0.,  1.]])
        self.T_ee_camera = np.identity(4)
        self.T_ee_camera[:3,:3] = R_ee_camera
        self.T_ee_camera[0:3, 3] = T_ee_camera_mm_translation / 1000.0
        rospy.loginfo(f"Using T_ee_camera:\n{self.T_ee_camera}")

        self.fixed_gripper_orientation_base_deg = rospy.get_param(
            '~fixed_gripper_orientation_base_deg', [0.0, 180.0, 45.0]
        )
        rospy.loginfo(f"Using fixed gripper orientation in BASE frame (degrees): {self.fixed_gripper_orientation_base_deg}")

        self.pick_z_offset_meters = rospy.get_param("~pick_z_offset_meters", 0.00)
        rospy.loginfo(f"Using pick Z offset from detected pick point: {self.pick_z_offset_meters} meters")
        
        self.place_z_offset_meters = rospy.get_param("~place_z_offset_meters", 0.00)
        rospy.loginfo(f"Using place Z offset from detected place point: {self.place_z_offset_meters} meters")

        self.pre_action_z_lift_meters = rospy.get_param("~pre_action_z_lift_meters", 0.05)
        rospy.loginfo(f"Using pre-action Z lift: {self.pre_action_z_lift_meters} meters")

        self.current_joint_angles_rad = None
        self.joint_names_from_driver = []

        self.joint_state_sub = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback, queue_size=1)
        rospy.loginfo("Subscribed to /base_feedback. Waiting for initial joint states...")
        rate = rospy.Rate(10)
        timeout_counter = 0
        while self.current_joint_angles_rad is None and not rospy.is_shutdown() and timeout_counter < 50:
            rate.sleep(); timeout_counter += 1
        if self.current_joint_angles_rad is None and not rospy.is_shutdown():
            rospy.logwarn(f"Timeout for BaseCyclic_Feedback. Trying sensor_msgs/JointState.")
            self.joint_state_sub.unregister()
            self.joint_names_from_driver = rospy.get_param(f"/{self.robot_name}/joint_names",
                                               [f"joint_{i+1}" for i in range(self.fk_calculator.DoF)])
            rospy.loginfo(f"Using joint names for JointState: {self.joint_names_from_driver}")
            self.joint_state_sub = rospy.Subscriber(
                f"/{self.robot_name}/joint_states", JointState, self.joint_states_callback, queue_size=1)
            rospy.loginfo(f"Subscribed to /{self.robot_name}/joint_states.")
            timeout_counter = 0
            while self.current_joint_angles_rad is None and not rospy.is_shutdown() and timeout_counter < 50:
                rate.sleep(); timeout_counter +=1
        if self.current_joint_angles_rad is None:
             rospy.logerr("CRITICAL: Could not receive joint states. Node will not function correctly.")
             self.is_arm_ready = False; return

        self.pick_and_place_sub = rospy.Subscriber(
            "kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera,
            self.pick_and_place_callback,
            queue_size=1
        )
        rospy.loginfo("Kinova Pick and Place Controller initialized. Waiting for goals on /kinova_pick_place/goal_in_camera")

    def base_feedback_callback(self, msg: BaseCyclic_Feedback):
        if hasattr(self.fk_calculator, 'DoF') and len(msg.actuators) >= self.fk_calculator.DoF:
            angles = [np.deg2rad(msg.actuators[i].position) for i in range(self.fk_calculator.DoF)]
            self.current_joint_angles_rad = angles
        elif not hasattr(self.fk_calculator, 'DoF'):
             rospy.logwarn_throttle(5, "FKCalculator object has no DoF attribute in base_feedback_callback.")
        else:
            rospy.logwarn_throttle(5, f"Base_feedback actuators: {len(msg.actuators)}, expected {self.fk_calculator.DoF}")


    def joint_states_callback(self, msg: JointState):
        if not self.joint_names_from_driver: rospy.logwarn_throttle(5, "Joint names order not set."); return
        angles_dict = dict(zip(msg.name, msg.position))
        try:
            self.current_joint_angles_rad = [angles_dict[name] for name in self.joint_names_from_driver]
        except KeyError as e:
            rospy.logwarn_throttle(5, f"Joint name '{e}' not found. Available: {list(angles_dict.keys())}")
            self.current_joint_angles_rad = None

    def get_current_T_base_camera(self):
        if self.current_joint_angles_rad is None: rospy.logerr("Current joint angles not available."); return None
        # 确保 fk_calculator 有 DoF 属性，并且与 current_joint_angles_rad 长度匹配
        if not hasattr(self.fk_calculator, 'DoF') or len(self.current_joint_angles_rad) != self.fk_calculator.DoF:
            rospy.logerr(f"FKCalculator DoF mismatch or not set. FK_DoF: {getattr(self.fk_calculator, 'DoF', 'Not Set')}, Angles_Len: {len(self.current_joint_angles_rad)}")
            return None
        self.fk_calculator.set_target_theta(self.current_joint_angles_rad, is_Deg=False)
        T_base_to_ee_mm = self.fk_calculator.T_build()
        T_base_to_ee_m = T_base_to_ee_mm.copy()
        T_base_to_ee_m[0:3, 3] /= 1000.0
        T_base_camera = T_base_to_ee_m @ self.T_ee_camera
        return T_base_camera

    def _transform_point_camera_to_base(self, point_in_camera_msg, T_base_camera_current):
        P_camera_homogeneous = np.array([point_in_camera_msg.x,
                                         point_in_camera_msg.y,
                                         point_in_camera_msg.z,
                                         1.0]).reshape(4, 1)
        P_base_homogeneous = T_base_camera_current @ P_camera_homogeneous
        return P_base_homogeneous[0:3, 0]

    def _go_to_home_safe_position(self):
        """尝试将机械臂移动到预定义的home/safe位置。"""
        rospy.loginfo("Attempting to move to home/safe position...")
        if not self.arm_controller.example_home_the_robot():
            rospy.logwarn("Failed to move to home/safe position.")
            # 即使失败，也尝试清除故障，为后续手动恢复做准备
            self.arm_controller.clear_robot_faults()
        else:
            rospy.loginfo("Successfully moved to home/safe position.")


    def _perform_action_sequence(self, action_name, target_xyz_base, z_offset_specific, is_pick_action):
        rospy.loginfo(f"--- Starting {action_name} Sequence ---")
        actual_gripper_target_z_base = target_xyz_base[2] + z_offset_specific
        target_orient_base_deg = self.fixed_gripper_orientation_base_deg
        pre_action_pos_x = target_xyz_base[0]
        pre_action_pos_y = target_xyz_base[1]
        pre_action_pos_z = actual_gripper_target_z_base + self.pre_action_z_lift_meters

        rospy.loginfo(f"Step 1 ({action_name}): Moving to pre-{action_name.lower()} position (X:{pre_action_pos_x:.3f}, Y:{pre_action_pos_y:.3f}, Z:{pre_action_pos_z:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to move to pre-{action_name.lower()} position. Aborting {action_name}.")
            self.arm_controller.clear_robot_faults(); return False

        if not is_pick_action:
            rospy.loginfo(f"Step 2 ({action_name}): Approaching precise place Z...")
        elif self.arm_controller.is_gripper_present:
            rospy.loginfo(f"Step 2 ({action_name}): Opening gripper...")
            rospy.sleep(1.0) # 确保手臂稳定后再操作夹爪
            if not self.arm_controller.move_gripper(0.0): rospy.logwarn("Failed to open gripper.")
            rospy.sleep(1.0) # 等待夹爪张开

        rospy.loginfo(f"Step 3 ({action_name}): Moving to precise {action_name.lower()} Z position (X:{target_xyz_base[0]:.3f}, Y:{target_xyz_base[1]:.3f}, Z:{actual_gripper_target_z_base:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            target_xyz_base[0], target_xyz_base[1], actual_gripper_target_z_base,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to move to {action_name.lower()} position. Aborting {action_name}.")
            self.arm_controller.clear_robot_faults(); return False
        rospy.sleep(0.5)

        if self.arm_controller.is_gripper_present:
            action_description = "Closing" if is_pick_action else "Opening"
            gripper_target_value = rospy.get_param("~grasp_closure_percentage", 0.8) if is_pick_action else 0.2 # 放置时稍微张开一些
            
            rospy.loginfo(f"Step 4 ({action_name}): {action_description} gripper to {gripper_target_value*100:.0f}%...")
            rospy.sleep(0.5) # 确保手臂稳定
            if not self.arm_controller.move_gripper(gripper_target_value): rospy.logwarn(f"Failed to {action_description.lower()} gripper.")
            rospy.sleep(1.5)

        rospy.loginfo(f"Step 5 ({action_name}): Lifting/Retreating from {action_name.lower()} point...")
        if not self.arm_controller.move_to_cartesian_pose(
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to lift/retreat from {action_name.lower()} point.")
            self.arm_controller.clear_robot_faults(); return False
        
        rospy.loginfo(f"--- {action_name} Sequence Completed ---")
        return True

    def pick_and_place_callback(self, msg: PickAndPlaceGoalInCamera):
        if not self.is_arm_ready or self.current_joint_angles_rad is None:
            rospy.logwarn("Arm not ready or joint angles unavailable, skipping pick and place command.")
            return

        rospy.loginfo(f"Received pick and place goal: Pick '{msg.object_id_at_pick}' at CamCoord, Place at '{msg.target_location_id_at_place}' at CamCoord.")
        T_base_camera_snapshot = self.get_current_T_base_camera()
        if T_base_camera_snapshot is None:
            rospy.logerr("Failed to get T_base_camera at the start of operation. Cannot proceed.")
            self._go_to_home_safe_position() # 尝试回到home位
            return

        rospy.loginfo(f"  Using T_base_camera snapshot for this operation:\n{T_base_camera_snapshot}")
        pick_point_base_xyz = self._transform_point_camera_to_base(msg.pick_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Pick Point in Base Frame: {pick_point_base_xyz}")
        place_point_base_xyz = self._transform_point_camera_to_base(msg.place_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Place Point in Base Frame: {place_point_base_xyz}")

        pick_successful = self._perform_action_sequence(
            action_name="Pick",
            target_xyz_base=pick_point_base_xyz,
            z_offset_specific=self.pick_z_offset_meters,
            is_pick_action=True
        )

        if not pick_successful:
            rospy.logwarn("Pick sequence failed. Aborting pick and place operation.")
            self._go_to_home_safe_position() # 尝试回到home位
            return

        place_successful = self._perform_action_sequence(
            action_name="Place",
            target_xyz_base=place_point_base_xyz,
            z_offset_specific=self.place_z_offset_meters,
            is_pick_action=False
        )

        if not place_successful:
            rospy.logwarn("Place sequence failed.")
            # 即使放置失败，也尝试回到home位
        # else: # 只有当放置也成功时，才打印操作完成
        #     rospy.loginfo("Pick and place operation completed successfully.")
        
        # 无论放置是否成功，在尝试完放置后都回到 home 位置
        rospy.loginfo("Pick and place attempt finished. Returning to home position.")
        self._go_to_home_safe_position()


    def run(self):
        if self.is_arm_ready:
            rospy.loginfo(f"{rospy.get_name()} is running and waiting for commands...")
            rospy.spin()
        else:
            rospy.logerr(f"{rospy.get_name()} could not start because arm is not ready or joint states unavailable.")


if __name__ == '__main__':
    try:
        controller_node = KinovaPickAndPlaceController()
        controller_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Kinova Pick and Place Controller interrupted.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()