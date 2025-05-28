#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

# --- 导入自定义消息 ---
from ee368_project.msg import TargetPositionInCamera 

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

class KinovaSimpleGraspController: 
    def __init__(self):
        rospy.init_node('kinova_simple_grasp_controller', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")
        rospy.loginfo(f"KinovaSimpleGraspController using robot_name: {self.robot_name}")

        self.arm_controller = SimplifiedArmController()
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

        # --- 预定义的末端执行器目标姿态 (在基座标系下) ---
        # 例如，夹爪垂直向下。theta_x, theta_y, theta_z (度)
        # 常见的是 [180, 0, X] 其中 X 取决于夹爪的初始安装和期望的工具x轴方向。
        # [180, 0, 90] 通常意味着工具Z轴向下, 工具X轴指向基座Y轴方向 (对于Kinova ZYX欧拉角)
        self.fixed_gripper_orientation_base_deg = rospy.get_param(
            '~fixed_gripper_orientation_base_deg', [180.0, 0.0, 90.0]
        )
        rospy.loginfo(f"Using fixed gripper orientation in BASE frame (degrees): {self.fixed_gripper_orientation_base_deg}")

        # --- 抓取时的Z轴偏移 (相对于棋子表面) ---
        # 正值表示在棋子中心点上方，负值表示在棋子中心点下方（如果棋子有厚度，可能需要微调）
        # 对于平放在平面上的棋子，可能希望稍微低于检测到的棋子中心Z值 (如果检测到的是棋子顶面)
        # 或者如果检测到的是棋子底面与棋盘接触点，则希望高于该点。
        # 假设检测到的是棋子顶面中心，我们希望夹爪略低于此，或正好在此高度。
        self.grasp_z_offset_from_detected_point_meters = rospy.get_param("~grasp_z_offset_meters", 0.00) # 例如，0.0表示尝试在检测到的Z值抓取
        rospy.loginfo(f"Using grasp Z offset from detected point: {self.grasp_z_offset_from_detected_point_meters} meters")


        self.current_joint_angles_rad = None
        self.joint_names_from_driver = []

        self.joint_state_sub = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback, queue_size=1)
        rospy.loginfo("Subscribed to /base_feedback. Waiting for initial joint states...")
        rate = rospy.Rate(10)
        timeout_counter = 0
        while self.current_joint_angles_rad is None and not rospy.is_shutdown() and timeout_counter < 50:
            rate.sleep()
            timeout_counter += 1
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
                rate.sleep()
                timeout_counter +=1
        if self.current_joint_angles_rad is None:
             rospy.logerr("CRITICAL: Could not receive joint states. Node will not function correctly.")
             self.is_arm_ready = False
             return

        self.target_position_sub = rospy.Subscriber(
            "kinova_grasp/target_position_in_camera", # 新的话题名称
            TargetPositionInCamera,                  # 新的消息类型
            self.target_position_callback,
            queue_size=1
        )
        rospy.loginfo("Kinova Simple Grasp Controller initialized. Waiting for target position commands on /kinova_grasp/target_position_in_camera")

    def base_feedback_callback(self, msg: BaseCyclic_Feedback):
        if len(msg.actuators) >= self.fk_calculator.DoF:
            angles = []
            for i in range(self.fk_calculator.DoF):
                angles.append(np.deg2rad(msg.actuators[i].position))
            self.current_joint_angles_rad = angles
        else:
            rospy.logwarn_throttle(5, f"Received base_feedback with {len(msg.actuators)} actuators, expected {self.fk_calculator.DoF}")

    def joint_states_callback(self, msg: JointState):
        if not self.joint_names_from_driver:
            rospy.logwarn_throttle(5, "Joint names order not set for joint_states_callback.")
            return
        angles_dict = dict(zip(msg.name, msg.position))
        ordered_angles = []
        try:
            for name in self.joint_names_from_driver:
                ordered_angles.append(angles_dict[name])
            self.current_joint_angles_rad = ordered_angles
        except KeyError as e:
            rospy.logwarn_throttle(5, f"Joint name '{e}' not found. Available: {list(angles_dict.keys())}")
            self.current_joint_angles_rad = None

    def get_current_T_base_camera(self):
        if self.current_joint_angles_rad is None:
            rospy.logerr("Current joint angles not available.")
            return None
        self.fk_calculator.set_target_theta(self.current_joint_angles_rad, is_Deg=False)
        T_base_to_ee = self.fk_calculator.T_build()
        T_base_camera = T_base_to_ee @ self.T_ee_camera
        return T_base_camera

    def target_position_callback(self, msg: TargetPositionInCamera):
        if not self.is_arm_ready or self.current_joint_angles_rad is None:
            rospy.logwarn("Arm not ready or joint angles unavailable, skipping command.")
            return

        rospy.loginfo(f"Received target position for '{msg.object_id}' in camera frame.")

        # 1. 目标点在相机坐标系中的齐次坐标 (z=0代表在相机xy平面，这不影响变换)
        P_camera_target_center = np.array([msg.position_in_camera.x,
                                           msg.position_in_camera.y,
                                           msg.position_in_camera.z,
                                           1.0]).reshape(4, 1)

        # 2. 计算当前的 T_base_camera
        T_base_camera_current = self.get_current_T_base_camera()
        if T_base_camera_current is None:
            rospy.logerr("Failed to get T_base_camera. Cannot transform target point.")
            return

        # 3. 将相机坐标系下的目标点转换到基座标系
        P_base_target_center_homogeneous = T_base_camera_current @ P_camera_target_center
        
        # 提取转换后的棋子中心点XYZ (在基座标系下)
        target_x_base_piece_center = P_base_target_center_homogeneous[0, 0]
        target_y_base_piece_center = P_base_target_center_homogeneous[1, 0]
        target_z_base_piece_center = P_base_target_center_homogeneous[2, 0]

        # 4. 计算实际的夹爪目标点 (考虑Z轴偏移)
        # 假设检测到的是棋子顶面中心，我们根据 grasp_z_offset_from_detected_point_meters 调整Z
        actual_gripper_target_z_base = target_z_base_piece_center + self.grasp_z_offset_from_detected_point_meters

        rospy.loginfo(f"  Target center in Cam: {P_camera_target_center[:3].flatten()}")
        rospy.loginfo(f"  Cam in Base (T_bc_cur):\n{T_base_camera_current}")
        rospy.loginfo(f"  Target center in Base: X={target_x_base_piece_center:.4f}, Y={target_y_base_piece_center:.4f}, Z(piece_center)={target_z_base_piece_center:.4f}")
        rospy.loginfo(f"  Gripper target Z in Base (after offset): Z(gripper)={actual_gripper_target_z_base:.4f}")


        # 5. 使用固定的末端执行器姿态
        target_orient_base_deg = self.fixed_gripper_orientation_base_deg

        rospy.loginfo(f"  Final command to arm (base frame): Pos=[{target_x_base_piece_center:.4f}, {target_y_base_piece_center:.4f}, {actual_gripper_target_z_base:.4f}], "
                      f"Orient(deg)=[{target_orient_base_deg[0]:.2f}, {target_orient_base_deg[1]:.2f}, {target_orient_base_deg[2]:.2f}]")

        # --- 执行抓取序列 (与上一版类似，但使用计算出的 actual_gripper_target_z_base) ---
        pre_grasp_z_offset_above_target = 0.05 # m, 在最终抓取Z值基础上再高5cm
        
        # A. 预抓取位置
        pre_grasp_pos_base_x = target_x_base_piece_center
        pre_grasp_pos_base_y = target_y_base_piece_center
        pre_grasp_pos_base_z = actual_gripper_target_z_base + pre_grasp_z_offset_above_target

        rospy.loginfo("Step 1: Moving to pre-grasp position...")
        success = self.arm_controller.move_to_cartesian_pose(
            pre_grasp_pos_base_x, pre_grasp_pos_base_y, pre_grasp_pos_base_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]
        )
        if not success:
            rospy.logwarn("Failed to move to pre-grasp position. Aborting grasp.")
            self.arm_controller.clear_robot_faults()
            return

        # B. （如果需要）打开夹爪
        if self.arm_controller.is_gripper_present:
            rospy.loginfo("Step 2: Opening gripper...")
            if not self.arm_controller.move_gripper(0.0): # 0.0 for fully open
                rospy.logwarn("Failed to open gripper.")
            rospy.sleep(1.0)

        # C. 移动到精确抓取位置 (使用 actual_gripper_target_z_base)
        rospy.loginfo("Step 3: Moving to grasp position...")
        success = self.arm_controller.move_to_cartesian_pose(
            target_x_base_piece_center, target_y_base_piece_center, actual_gripper_target_z_base,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]
        )
        if not success:
            rospy.logwarn("Failed to move to grasp position. Aborting grasp.")
            self.arm_controller.clear_robot_faults()
            return
        rospy.sleep(0.5)

        # D. 关闭夹爪
        if self.arm_controller.is_gripper_present:
            rospy.loginfo("Step 4: Closing gripper...")
            grasp_closure_percentage = rospy.get_param("~grasp_closure_percentage", 0.7)
            if not self.arm_controller.move_gripper(grasp_closure_percentage):
                rospy.logwarn("Failed to close gripper.")
            rospy.sleep(1.5)

        # E. 向上提起物体
        rospy.loginfo("Step 5: Lifting object...")
        # 提升回预抓取时的Z高度
        success = self.arm_controller.move_to_cartesian_pose(
            pre_grasp_pos_base_x, pre_grasp_pos_base_y, pre_grasp_pos_base_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]
        )
        if success:
            rospy.loginfo("Grasp sequence potentially successful (object lifted).")
        else:
            rospy.logwarn("Failed to lift object.")
            self.arm_controller.clear_robot_faults()


    def run(self):
        if self.is_arm_ready:
            rospy.loginfo(f"{rospy.get_name()} is running and waiting for commands...")
            rospy.spin()
        else:
            rospy.logerr(f"{rospy.get_name()} could not start because arm is not ready or joint states unavailable.")


if __name__ == '__main__':
    try:
        controller_node = KinovaSimpleGraspController()
        controller_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Kinova Simple Grasp Controller interrupted.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()