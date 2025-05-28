#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

# --- 导入自定义消息 ---
# from ee368_project.msg import TargetPositionInCamera # 旧的
from ee368_project.msg import PickAndPlaceGoalInCamera # 新的 pick and place 消息

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

class KinovaPickAndPlaceController: # 重命名类以反映新功能
    def __init__(self):
        rospy.init_node('kinova_pick_and_place_controller', anonymous=False) # 新的节点名

        self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")
        rospy.loginfo(f"KinovaPickAndPlaceController using robot_name: {self.robot_name}")

        self.arm_controller = SimplifiedArmController(robot_name_param=self.robot_name) # 传递 robot_name
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
            '~fixed_gripper_orientation_base_deg', [180.0, 0.0, 90.0]
        )
        rospy.loginfo(f"Using fixed gripper orientation in BASE frame (degrees): {self.fixed_gripper_orientation_base_deg}")

        self.pick_z_offset_meters = rospy.get_param("~pick_z_offset_meters", 0.00)
        rospy.loginfo(f"Using pick Z offset from detected pick point: {self.pick_z_offset_meters} meters")
        
        self.place_z_offset_meters = rospy.get_param("~place_z_offset_meters", 0.00) # Z偏移用于放置
        rospy.loginfo(f"Using place Z offset from detected place point: {self.place_z_offset_meters} meters")

        self.pre_action_z_lift_meters = rospy.get_param("~pre_action_z_lift_meters", 0.05) # 预抓取/预放置的抬高量
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

        # 订阅新的 PickAndPlaceGoalInCamera 消息
        self.pick_and_place_sub = rospy.Subscriber(
            "kinova_pick_place/goal_in_camera", # 建议的话题名称
            PickAndPlaceGoalInCamera,
            self.pick_and_place_callback,
            queue_size=1
        )
        rospy.loginfo("Kinova Pick and Place Controller initialized. Waiting for goals on /kinova_pick_place/goal_in_camera")

    def base_feedback_callback(self, msg: BaseCyclic_Feedback):
        if len(msg.actuators) >= self.fk_calculator.DoF:
            angles = [np.deg2rad(msg.actuators[i].position) for i in range(self.fk_calculator.DoF)]
            self.current_joint_angles_rad = angles
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
        self.fk_calculator.set_target_theta(self.current_joint_angles_rad, is_Deg=False)
        T_base_to_ee_mm = self.fk_calculator.T_build()
        T_base_to_ee_m = T_base_to_ee_mm.copy()
        T_base_to_ee_m[0:3, 3] /= 1000.0
        T_base_camera = T_base_to_ee_m @ self.T_ee_camera
        return T_base_camera

    def _transform_point_camera_to_base(self, point_in_camera_msg, T_base_camera_current):
        """Helper function to transform a geometry_msgs/Point from camera to base frame."""
        P_camera_homogeneous = np.array([point_in_camera_msg.x,
                                         point_in_camera_msg.y,
                                         point_in_camera_msg.z,
                                         1.0]).reshape(4, 1)
        P_base_homogeneous = T_base_camera_current @ P_camera_homogeneous
        return P_base_homogeneous[0:3, 0] # 返回 (x,y,z) numpy array in base frame

    def _perform_action_sequence(self, action_name, target_xyz_base, z_offset_specific, is_pick_action):
        """
        Helper function to perform a sequence of movements for picking or placing.
        action_name: "Pick" or "Place" for logging.
        target_xyz_base: The (x,y,z) of the object's center in base frame (before specific z_offset).
        z_offset_specific: The pick_z_offset_meters or place_z_offset_meters.
        is_pick_action: Boolean, True if picking, False if placing.
        Returns: True if sequence seems successful, False otherwise.
        """
        rospy.loginfo(f"--- Starting {action_name} Sequence ---")

        actual_gripper_target_z_base = target_xyz_base[2] + z_offset_specific
        target_orient_base_deg = self.fixed_gripper_orientation_base_deg

        # A. 预动作位置 (在最终动作Z值基础上再抬高)
        pre_action_pos_x = target_xyz_base[0]
        pre_action_pos_y = target_xyz_base[1]
        pre_action_pos_z = actual_gripper_target_z_base + self.pre_action_z_lift_meters

        rospy.loginfo(f"Step 1 ({action_name}): Moving to pre-{action_name.lower()} position (X:{pre_action_pos_x:.3f}, Y:{pre_action_pos_y:.3f}, Z:{pre_action_pos_z:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to move to pre-{action_name.lower()} position. Aborting {action_name}.")
            self.arm_controller.clear_robot_faults(); return False

        # B. 如果是放置，先移动到精确放置点上方；如果是抓取，则打开夹爪
        if not is_pick_action: # Placing
            rospy.loginfo(f"Step 2 ({action_name}): Moving to precise place approach Z...") #保持在抬高Z，然后下降
            # 这一步是可选的，也可以直接在下一步下降
        elif self.arm_controller.is_gripper_present: # Picking
            rospy.loginfo(f"Step 2 ({action_name}): Opening gripper...")
            if not self.arm_controller.move_gripper(0.0): rospy.logwarn("Failed to open gripper.")
            rospy.sleep(1.0)

        # C. 移动到精确的抓取/放置Z高度
        rospy.loginfo(f"Step 3 ({action_name}): Moving to precise {action_name.lower()} Z position (X:{target_xyz_base[0]:.3f}, Y:{target_xyz_base[1]:.3f}, Z:{actual_gripper_target_z_base:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            target_xyz_base[0], target_xyz_base[1], actual_gripper_target_z_base,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to move to {action_name.lower()} position. Aborting {action_name}.")
            self.arm_controller.clear_robot_faults(); return False
        rospy.sleep(0.5)

        # D. 执行夹爪动作 (抓取时闭合，放置时打开)
        if self.arm_controller.is_gripper_present:
            if is_pick_action:
                rospy.loginfo(f"Step 4 ({action_name}): Closing gripper...")
                grasp_closure = rospy.get_param("~grasp_closure_percentage", 0.7)
                if not self.arm_controller.move_gripper(grasp_closure): rospy.logwarn("Failed to close gripper.")
            else: # Placing
                rospy.loginfo(f"Step 4 ({action_name}): Opening gripper...")
                if not self.arm_controller.move_gripper(0.0): rospy.logwarn("Failed to open gripper.")
            rospy.sleep(1.5) # 等待夹爪动作

        # E. 向上提起/离开
        rospy.loginfo(f"Step 5 ({action_name}): Lifting/Retreating from {action_name.lower()} point...")
        if not self.arm_controller.move_to_cartesian_pose( # 回到预动作的Z高度
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"Failed to lift/retreat from {action_name.lower()} point.")
            self.arm_controller.clear_robot_faults(); return False # 即使提起失败也返回False
        
        rospy.loginfo(f"--- {action_name} Sequence Completed ---")
        return True


    def pick_and_place_callback(self, msg: PickAndPlaceGoalInCamera):
        if not self.is_arm_ready or self.current_joint_angles_rad is None:
            rospy.logwarn("Arm not ready or joint angles unavailable, skipping pick and place command.")
            return

        rospy.loginfo(f"Received pick and place goal: Pick '{msg.object_id_at_pick}' at CamCoord, Place at '{msg.target_location_id_at_place}' at CamCoord.")

        # 关键：在整个pick-and-place操作开始前，获取一次当前的 T_base_camera
        # 所有的坐标变换都基于这个“快照”
        T_base_camera_snapshot = self.get_current_T_base_camera()
        if T_base_camera_snapshot is None:
            rospy.logerr("Failed to get T_base_camera at the start of operation. Cannot proceed.")
            return

        rospy.loginfo(f"  Using T_base_camera snapshot for this operation:\n{T_base_camera_snapshot}")

        # 1. 计算抓取点在基座标系下的坐标
        pick_point_base_xyz = self._transform_point_camera_to_base(msg.pick_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Pick Point in Base Frame: {pick_point_base_xyz}")

        # 2. 计算放置点在基座标系下的坐标
        place_point_base_xyz = self._transform_point_camera_to_base(msg.place_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  Calculated Place Point in Base Frame: {place_point_base_xyz}")

        # --- 执行抓取序列 ---
        pick_successful = self._perform_action_sequence(
            action_name="Pick",
            target_xyz_base=pick_point_base_xyz,
            z_offset_specific=self.pick_z_offset_meters,
            is_pick_action=True
        )

        if not pick_successful:
            rospy.logwarn("Pick sequence failed. Aborting pick and place operation.")
            # 可以在这里尝试将机械臂移动到一个安全位置
            # self.go_to_safe_home_position() # 你需要实现这个方法
            return

        # --- （可选）中间移动，如果抓取点和放置点之间需要特定的路径规划 ---
        # rospy.loginfo("Moving to an intermediate point (if necessary)...")
        # 简单的实现是直接从提起点移动到预放置点，如果距离远，可能需要更平滑的路径

        # --- 执行放置序列 ---
        place_successful = self._perform_action_sequence(
            action_name="Place",
            target_xyz_base=place_point_base_xyz,
            z_offset_specific=self.place_z_offset_meters,
            is_pick_action=False
        )

        if not place_successful:
            rospy.logwarn("Place sequence failed.")
            # 即使放置失败，也可能需要将机械臂移到安全位置
            # self.go_to_safe_home_position()
            return

        rospy.loginfo("Pick and place operation completed.")
        # self.go_to_safe_home_position() # 操作完成后回到待命位置


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