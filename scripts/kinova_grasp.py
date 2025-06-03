#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

# --- 导入自定义消息 ---
from ee368_project.msg import PickAndPlaceGoalInCamera
from std_msgs.msg import String # <<< 新增：用于发布机械臂状态

# --- 从同一包的scripts目录导入其他模块 ---
try:
    from move_cartesian import SimplifiedArmController
    from forward_kinematics import arm as FKCalculator
except ImportError as e:
    rospy.logerr(f"未能导入 SimplifiedArmController 或 FKCalculator: {e}")
    import sys
    sys.exit(1)

from sensor_msgs.msg import JointState
from kortex_driver.msg import BaseCyclic_Feedback

class KinovaPickAndPlaceController:
    # --- 机械臂状态常量 ---
    STATE_IDLE = 0
    STATE_BUSY = 1
    STATE_ERROR = 2

    # --- 机械臂状态对应的发布消息 ---
    MSG_ARM_IDLE = "0"
    MSG_ARM_BUSY = "1"
    MSG_ARM_ERROR = "2"

    def __init__(self):
        rospy.init_node('kinova_pick_and_place_controller', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")
        rospy.loginfo(f"KinovaPickAndPlaceController 使用机器人名称: {self.robot_name}")

        # --- 机械臂状态发布器 ---
        self.arm_status_pub = rospy.Publisher(
            f"/{self.robot_name}/arm_status", String, queue_size=1, latch=True
        )
        self.current_arm_state = None # 初始化当前机械臂内部状态
        # ---

        self.arm_controller = SimplifiedArmController(robot_name_param=self.robot_name)
        self.is_arm_ready = False

        if not self.arm_controller.is_init_success:
            rospy.logerr("机械臂控制器初始化失败。")
            self._update_and_publish_arm_status(self.STATE_ERROR) # <<< 设置状态
            return
        if self.arm_controller.clear_robot_faults() and self.arm_controller.activate_notifications():
            self.is_arm_ready = True
            rospy.loginfo("机械臂已就绪并已激活通知。")
        else:
            rospy.logerr("初始化机械臂失败。")
            self._update_and_publish_arm_status(self.STATE_ERROR) # <<< 设置状态
            return

        self.fk_calculator = FKCalculator(Dof=6) # 假设使用6自由度

        T_ee_camera_mm_translation = np.array([60.0, -40.0, -110.0])
        R_ee_camera = np.array([[0., -1.,  0.], [1.,  0.,  0.], [0.,  0.,  1.]])
        self.T_ee_camera = np.identity(4)
        self.T_ee_camera[:3,:3] = R_ee_camera
        self.T_ee_camera[0:3, 3] = T_ee_camera_mm_translation / 1000.0
        rospy.loginfo(f"使用的 T_ee_camera (末端到相机变换矩阵):\n{self.T_ee_camera}")

        self.fixed_gripper_orientation_base_deg = rospy.get_param(
            '~fixed_gripper_orientation_base_deg', [0.0, 180.0, 45.0]
        )
        rospy.loginfo(f"使用的固定夹爪姿态 (基座标系, 度): {self.fixed_gripper_orientation_base_deg}")

        self.pick_z_offset_meters = rospy.get_param("~pick_z_offset_meters", 0.00)
        rospy.loginfo(f"使用的抓取Z轴偏移 (从检测到的抓取点): {self.pick_z_offset_meters} 米")
        
        self.place_z_offset_meters = rospy.get_param("~place_z_offset_meters", 0.00)
        rospy.loginfo(f"使用的放置Z轴偏移 (从检测到的放置点): {self.place_z_offset_meters} 米")

        self.pre_action_z_lift_meters = rospy.get_param("~pre_action_z_lift_meters", 0.05)
        rospy.loginfo(f"使用的动作前Z轴抬升高度: {self.pre_action_z_lift_meters} 米")

        self.current_joint_angles_rad = None
        self.joint_names_from_driver = []

        self.joint_state_sub = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback, queue_size=1)
        rospy.loginfo("已订阅 /base_feedback。等待初始关节状态...")
        rate = rospy.Rate(10)
        timeout_counter = 0
        while self.current_joint_angles_rad is None and not rospy.is_shutdown() and timeout_counter < 50:
            rate.sleep(); timeout_counter += 1
        
        if self.current_joint_angles_rad is None and not rospy.is_shutdown():
            rospy.logwarn(f"获取 BaseCyclic_Feedback 超时。尝试 sensor_msgs/JointState。")
            self.joint_state_sub.unregister()
            self.joint_names_from_driver = rospy.get_param(f"/{self.robot_name}/joint_names", 
                                               [f"joint_{i+1}" for i in range(self.fk_calculator.DoF)])
            rospy.loginfo(f"用于 JointState 的关节名称: {self.joint_names_from_driver}")
            self.joint_state_sub = rospy.Subscriber(
                f"/{self.robot_name}/joint_states", JointState, self.joint_states_callback, queue_size=1)
            rospy.loginfo(f"已订阅 /{self.robot_name}/joint_states。")
            timeout_counter = 0
            while self.current_joint_angles_rad is None and not rospy.is_shutdown() and timeout_counter < 50:
                rate.sleep(); timeout_counter +=1

        if self.current_joint_angles_rad is None:
             rospy.logerr("严重错误: 无法接收到关节状态。节点将无法正常工作。")
             self.is_arm_ready = False
             self._update_and_publish_arm_status(self.STATE_ERROR) # <<< 设置状态
             return

        self.pick_and_place_sub = rospy.Subscriber(
            "kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera,
            self.pick_and_place_callback,
            queue_size=1
        )
        rospy.loginfo("Kinova 抓取和放置控制器已初始化。等待 /kinova_pick_place/goal_in_camera 上的目标...")

        # --- 初始化归位和状态设置 ---
        if self.is_arm_ready:
            rospy.loginfo("执行初始化归位序列...")
            self._go_to_home_safe_position() # 此函数会在成功后设置状态为待命
        # ---

    def _update_and_publish_arm_status(self, new_state):
        """辅助函数，用于更新并发布机械臂状态（如果状态发生变化）。"""
        if new_state != self.current_arm_state:
            self.current_arm_state = new_state
            status_msg_str = ""
            if new_state == self.STATE_IDLE:
                status_msg_str = self.MSG_ARM_IDLE
            elif new_state == self.STATE_BUSY:
                status_msg_str = self.MSG_ARM_BUSY
            elif new_state == self.STATE_ERROR:
                status_msg_str = self.MSG_ARM_ERROR
            else:
                rospy.logwarn(f"未知的机械臂状态代码: {new_state}")
                status_msg_str = f"未知状态 ({new_state})" # Fallback message

            rospy.loginfo(f"机械臂状态变更为: {status_msg_str}")
            self.arm_status_pub.publish(String(data=status_msg_str))

    def base_feedback_callback(self, msg: BaseCyclic_Feedback):
        if hasattr(self.fk_calculator, 'DoF') and len(msg.actuators) >= self.fk_calculator.DoF:
            angles = [np.deg2rad(msg.actuators[i].position) for i in range(self.fk_calculator.DoF)]
            self.current_joint_angles_rad = angles
        elif not hasattr(self.fk_calculator, 'DoF'):
             rospy.logwarn_throttle(5, "FKCalculator 对象在 base_feedback_callback 中没有 DoF 属性。")
        else:
            rospy.logwarn_throttle(5, f"Base_feedback 中的执行器数量: {len(msg.actuators)}, 期望: {self.fk_calculator.DoF}")

    def joint_states_callback(self, msg: JointState):
        if not self.joint_names_from_driver: rospy.logwarn_throttle(5, "关节名称顺序未设置。"); return
        angles_dict = dict(zip(msg.name, msg.position))
        try:
            self.current_joint_angles_rad = [angles_dict[name] for name in self.joint_names_from_driver]
        except KeyError as e:
            rospy.logwarn_throttle(5, f"关节名称 '{e}' 未找到。可用名称: {list(angles_dict.keys())}")
            self.current_joint_angles_rad = None

    def get_current_T_base_camera(self):
        if self.current_joint_angles_rad is None: rospy.logerr("当前关节角度不可用。"); return None
        if not hasattr(self.fk_calculator, 'DoF') or len(self.current_joint_angles_rad) != self.fk_calculator.DoF:
            rospy.logerr(f"FKCalculator DoF 不匹配或未设置. FK_DoF: {getattr(self.fk_calculator, 'DoF', '未设置')}, 角度数量: {len(self.current_joint_angles_rad)}")
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
        """尝试将机械臂移动到预定义的home/safe位置，并更新状态。"""
        rospy.loginfo("尝试移动到 home/safe 位置...")
        self._update_and_publish_arm_status(self.STATE_BUSY) # <<< 机械臂即将移动

        if not self.arm_controller.example_home_the_robot():
            rospy.logwarn("移动到 home/safe 位置失败。")
            # 状态保持繁忙 (STATE_BUSY)，因为它不在可靠的 home/idle 状态。
            self.arm_controller.clear_robot_faults()
        else:
            rospy.loginfo("成功移动到 home/safe 位置。")
            self._update_and_publish_arm_status(self.STATE_IDLE) # <<< 机械臂现在处于待命状态

    def _perform_action_sequence(self, action_name, target_xyz_base, z_offset_specific, is_pick_action):
        rospy.loginfo(f"--- 开始 {action_name} 序列 ---")
        actual_gripper_target_z_base = target_xyz_base[2] + z_offset_specific
        target_orient_base_deg = self.fixed_gripper_orientation_base_deg
        pre_action_pos_x = target_xyz_base[0]
        pre_action_pos_y = target_xyz_base[1]
        pre_action_pos_z = actual_gripper_target_z_base + self.pre_action_z_lift_meters

        rospy.loginfo(f"步骤 1 ({action_name}): 移动到预{action_name.lower()}位置 (X:{pre_action_pos_x:.3f}, Y:{pre_action_pos_y:.3f}, Z:{pre_action_pos_z:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"移动到预{action_name.lower()}位置失败。正在中止 {action_name}。")
            self.arm_controller.clear_robot_faults(); return False

        if not is_pick_action:
            rospy.loginfo(f"步骤 2 ({action_name}): 接近精确的放置Z值...")
        elif self.arm_controller.is_gripper_present:
            rospy.loginfo(f"步骤 2 ({action_name}): 为抓取打开夹爪至70%...")
            rospy.sleep(1.0)
            if not self.arm_controller.move_gripper(0.6): rospy.logwarn("打开夹爪至70%失败。")
            rospy.sleep(1.0)

        rospy.loginfo(f"步骤 3 ({action_name}): 移动到精确的{action_name.lower()}Z位置 (X:{target_xyz_base[0]:.3f}, Y:{target_xyz_base[1]:.3f}, Z:{actual_gripper_target_z_base:.3f})...")
        if not self.arm_controller.move_to_cartesian_pose(
            target_xyz_base[0], target_xyz_base[1], actual_gripper_target_z_base,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"移动到{action_name.lower()}位置失败。正在中止 {action_name}。")
            self.arm_controller.clear_robot_faults(); return False
        rospy.sleep(0.5)

        if self.arm_controller.is_gripper_present:
            action_description = "闭合" if is_pick_action else "打开"
            gripper_target_value = rospy.get_param("~grasp_closure_percentage", 0.75) if is_pick_action else 0.6 
            
            rospy.loginfo(f"步骤 4 ({action_name}): {action_description}夹爪至 {gripper_target_value*100:.0f}%...")
            rospy.sleep(0.5)
            if not self.arm_controller.move_gripper(gripper_target_value): rospy.logwarn(f"{action_description}夹爪失败。")
            rospy.sleep(1.5)

        rospy.loginfo(f"步骤 5 ({action_name}): 从{action_name.lower()}点抬升/撤退...")
        if not self.arm_controller.move_to_cartesian_pose(
            pre_action_pos_x, pre_action_pos_y, pre_action_pos_z,
            target_orient_base_deg[0], target_orient_base_deg[1], target_orient_base_deg[2]):
            rospy.logwarn(f"从{action_name.lower()}点抬升/撤退失败。")
            self.arm_controller.clear_robot_faults(); return False
        
        rospy.loginfo(f"--- {action_name} 序列完成 ---")
        return True

    def pick_and_place_callback(self, msg: PickAndPlaceGoalInCamera):
        if self.current_arm_state == self.STATE_BUSY and self.is_arm_ready:
            rospy.logwarn(f"机械臂当前状态为 '{self.MSG_ARM_BUSY}'。忽略新的抓取放置目标: '{msg.object_id_at_pick}'。")
            return

        if not self.is_arm_ready or self.current_joint_angles_rad is None:
            rospy.logwarn("机械臂未就绪或关节角度不可用，跳过抓取和放置指令。")
            if not self.is_arm_ready:
                 self._update_and_publish_arm_status(self.STATE_ERROR)
            return
        
        self._update_and_publish_arm_status(self.STATE_BUSY)

        rospy.loginfo(f"收到抓取和放置目标: 抓取 '{msg.object_id_at_pick}' (相机坐标系), 放置到 '{msg.target_location_id_at_place}' (相机坐标系)。")
        T_base_camera_snapshot = self.get_current_T_base_camera()
        if T_base_camera_snapshot is None:
            rospy.logerr("操作开始时未能获取 T_base_camera。无法继续。")
            self._go_to_home_safe_position() 
            return

        rospy.loginfo(f"  本次操作使用的 T_base_camera 快照:\n{T_base_camera_snapshot}")
        pick_point_base_xyz = self._transform_point_camera_to_base(msg.pick_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  计算得到的抓取点 (基座标系): {pick_point_base_xyz}")
        place_point_base_xyz = self._transform_point_camera_to_base(msg.place_position_in_camera, T_base_camera_snapshot)
        rospy.loginfo(f"  计算得到的放置点 (基座标系): {place_point_base_xyz}")

        pick_successful = self._perform_action_sequence(
            action_name="抓取",
            target_xyz_base=pick_point_base_xyz,
            z_offset_specific=self.pick_z_offset_meters,
            is_pick_action=True
        )

        if not pick_successful:
            rospy.logwarn("抓取序列失败。中止抓取和放置操作。")
            self._go_to_home_safe_position() 
            return

        place_successful = self._perform_action_sequence(
            action_name="放置",
            target_xyz_base=place_point_base_xyz,
            z_offset_specific=self.place_z_offset_meters,
            is_pick_action=False
        )

        if not place_successful:
            rospy.logwarn("放置序列失败。")
        
        rospy.loginfo("抓取和放置尝试完成。正在返回到 home 位置。")
        self._go_to_home_safe_position()

    def run(self):
        if self.is_arm_ready:
            rospy.loginfo(f"{rospy.get_name()} 正在运行并等待指令...")
            rospy.spin()
        else:
            rospy.logerr(f"{rospy.get_name()} 无法启动，因为机械臂未就绪或关节状态不可用。")
            self._update_and_publish_arm_status(self.STATE_ERROR)

if __name__ == '__main__':
    try:
        controller_node = KinovaPickAndPlaceController()
        controller_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Kinova 抓取和放置控制器被中断。")
    except Exception as e:
        rospy.logerr(f"发生未处理的异常: {e}")
        import traceback
        traceback.print_exc()