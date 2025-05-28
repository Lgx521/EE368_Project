#!/usr/bin/env python3


import sys
import rospy
import time

from kortex_driver.srv import *
from kortex_driver.msg import *

class SimplifiedArmController:
    def __init__(self, robot_name_param="my_gen3_lite"):
        try:
            #rospy.init_node('simplified_arm_controller_python')

            # Get node params
            self.robot_name = robot_name_param
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo(f"Using robot_name {self.robot_name}, is_gripper_present: {self.is_gripper_present}")

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber(f"/{self.robot_name}/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = f'/{self.robot_name}/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            execute_action_full_name = f'/{self.robot_name}/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            if self.is_gripper_present:
                send_gripper_command_full_name = f'/{self.robot_name}/base/send_gripper_command'
                rospy.wait_for_service(send_gripper_command_full_name)
                self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            activate_publishing_of_action_notification_full_name = f'/{self.robot_name}/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

            self.is_init_success = True
        except Exception as e:
            rospy.logerr(f"Initialization failed: {e}")
            self.is_init_success = False

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def _fill_cartesian_waypoint(self, x, y, z, theta_x, theta_y, theta_z, blending_radius=0.0):
        """Helper function to create a CartesianWaypoint."""
        waypoint = Waypoint()
        cartesian_waypoint = CartesianWaypoint()

        cartesian_waypoint.pose.x = x
        cartesian_waypoint.pose.y = y
        cartesian_waypoint.pose.z = z
        cartesian_waypoint.pose.theta_x = theta_x  # In degrees
        cartesian_waypoint.pose.theta_y = theta_y  # In degrees
        cartesian_waypoint.pose.theta_z = theta_z  # In degrees
        # Reference frame is typically BASE for absolute movements
        cartesian_waypoint.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        cartesian_waypoint.blending_radius = blending_radius # 0 for stop at waypoint
        waypoint.oneof_type_of_waypoint.cartesian_waypoint.append(cartesian_waypoint)

        return waypoint

    def _wait_for_action_end_or_abort(self):
        """Waits for an action to complete or abort."""
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.logwarn("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)
        return False # rospy.is_shutdown()

    def activate_notifications(self):
        """Activates action notifications from the robot."""
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
            rospy.loginfo("Successfully activated Action Notifications!")
            rospy.sleep(1.0) # Give it a moment to register
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call OnNotificationActionTopic: {e}")
            return False

    def clear_robot_faults(self):
        """Clears any existing faults on the robot."""
        try:
            self.clear_faults()
            rospy.loginfo("Cleared faults successfully.")
            rospy.sleep(1.0) # Give it a moment
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call ClearFaults: {e}")
            return False

    def move_to_cartesian_pose(self, x, y, z, theta_x, theta_y, theta_z):
        """
        Moves the robot's end-effector to a specified absolute Cartesian pose.
        x, y, z: Target position in meters (relative to base frame).
        theta_x, theta_y, theta_z: Target orientation in degrees (Euler angles, typically Tait-Bryan ZYX convention for Kinova).
        """
        if not self.is_init_success:
            rospy.logerr("Controller not initialized. Cannot move.")
            return False

        self.last_action_notif_type = None # Reset before starting new action

        req = ExecuteActionRequest()
        trajectory = WaypointList()

        # Create a single waypoint for the target pose
        waypoint = self._fill_cartesian_waypoint(x, y, z, theta_x, theta_y, theta_z, 0.0)
        trajectory.waypoints.append(waypoint)

        # Duration 0 means the robot will use its default speed profile
        trajectory.duration = 0
        trajectory.use_optimal_blending = False # Not relevant for a single waypoint

        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)

        rospy.loginfo(f"Sending robot to Cartesian pose: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, ThX={theta_x:.1f}, ThY={theta_y:.1f}, ThZ={theta_z:.1f}")
        try:
            self.execute_action(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call ExecuteAction for Cartesian pose: {e}")
            return False
        else:
            rospy.loginfo("Cartesian pose command sent. Waiting for completion...")
            # return self._wait_for_action_end_or_abort()
            rospy.sleep(2)
            return True


    def move_gripper(self, position_percentage):
        """
        Moves the gripper to a specified position.
        position_percentage: 0.0 (fully open) to 1.0 (fully closed).
        """
        if not self.is_init_success:
            rospy.logerr("Controller not initialized. Cannot move gripper.")
            return False
        if not self.is_gripper_present:
            rospy.logwarn("No gripper is present on the arm. Command ignored.")
            return True # Or False, depending on how you want to handle this

        # Initialize the request
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0 # Usually 0 for the first (or only) finger/gripper mechanism
        finger.value = float(position_percentage) # Ensure it's a float
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo(f"Sending gripper to position: {position_percentage*100:.0f}%")

        try:
            self.send_gripper_command(req)
            time.sleep(0.75) # Allow time for gripper to move; no direct action notification for this simple command
            rospy.loginfo("Gripper command sent.")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call SendGripperCommand: {e}")
            return False
        except AttributeError:
            rospy.logerr("Gripper service proxy not initialized. Is gripper present and configured?")
            return False

    def example_home_the_robot(self):
        """Homes the robot using the predefined Home Action (ID #2)."""
        if not self.is_init_success:
            rospy.logerr("Controller not initialized. Cannot home.")
            return False
        
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        # This requires the ReadAction service, which we removed for strictness.
        # Re-adding it if Homing is desired.
        # For now, let's assume a "safe" or "retract" position instead of full API home.
        rospy.loginfo("Homing action (moving to a predefined safe pose) requested.")

        home_x, home_y, home_z = 0.30, 0.10, 0.30
        home_thx, home_thy, home_thz = 0.0, 180.0, 45.0 # Tool pointing down

        rospy.loginfo("Moving to a predefined 'home' Cartesian position.")
        return self.move_to_cartesian_pose(home_x, home_y, home_z, home_thx, home_thy, home_thz)
    

def go_to_cartesian_pos(pos, orientation=[0, 180, 45]):

    '''
    外部调用的控制机械臂运动的函数
    默认orientation为朝下
    pos=[pos[0], pos[1], pos[2]]，为目标的cartesian坐标位置
    '''

    controller = SimplifiedArmController()
    success = controller.is_init_success

    if success:
        success &= controller.clear_robot_faults()
        if not success:
            rospy.logerr("Failed to clear faults. Aborting.")
            return
        
        success &= controller.activate_notifications()
        if not success:
            rospy.logerr("Failed to activate notifications. Aborting.")
            return
        
        target_x = pos[0]
        target_y = pos[1]
        target_z = pos[2]
        target_theta_x = orientation[0]
        target_theta_y = orientation[1]
        target_theta_z = orientation[2]

        rospy.loginfo(f"--- Moving to target Cartesian pose 1 ({target_x}, {target_y}, {target_z}) ---")
        success &= controller.move_to_cartesian_pose(target_x, target_y, target_z, target_theta_x, target_theta_y, target_theta_z)
        if not success: rospy.logwarn(f"Move to pose 1 failed or was aborted.")


def main():
    controller = SimplifiedArmController()
    success = controller.is_init_success

    if success:
        # 1. Clear faults
        success &= controller.clear_robot_faults()
        if not success:
            rospy.logerr("Failed to clear faults. Aborting.")
            return

        # 2. Activate notifications
        success &= controller.activate_notifications()
        if not success:
            rospy.logerr("Failed to activate notifications. Aborting.")
            return

        # 3. Example Gripper Movement
        # if controller.is_gripper_present:
        #     rospy.loginfo("--- Opening gripper ---")
        #     success &= controller.move_gripper(0.0) # Fully open
        #     if not success: rospy.logwarn("Gripper open command failed or issue occurred.")
        #     rospy.sleep(2)

        #     rospy.loginfo("--- Closing gripper to 50% ---")
        #     success &= controller.move_gripper(0.5) # 50% closed
        #     if not success: rospy.logwarn("Gripper 50% close command failed or issue occurred.")
        #     rospy.sleep(2)
        # else:
        #     rospy.loginfo("Skipping gripper commands as no gripper is present.")


        # 4. Example Cartesian Movement
        # IMPORTANT: Define SAFE and REACHABLE Cartesian coordinates for your robot and environment!
        # These are just placeholder values.
        # X, Y, Z are in meters from the robot base.
        # ThetaX, ThetaY, ThetaZ are in degrees (Euler angles, often Tait-Bryan ZYX).
        target_x = 0.3
        target_y = 0.1
        target_z = 0.3
        target_theta_x = 0 # Example: tool pointing down
        target_theta_y = 180.0
        target_theta_z = 45

        rospy.loginfo(f"--- Moving to target Cartesian pose 1 ({target_x}, {target_y}, {target_z}) ---")
        success &= controller.move_to_cartesian_pose(target_x, target_y, target_z, target_theta_x, target_theta_y, target_theta_z)
        if not success: rospy.logwarn(f"Move to pose 1 failed or was aborted.")
        rospy.sleep(1)

        # Example of a second pose
        # target_x_2 = 0.312
        # target_y_2 = 0.055
        # target_z_2 = -0.059
        target_x_2 = 0.363
        target_y_2 = 0.238
        target_z_2 = -0.01
        # Keep orientation same or change as needed
        target_theta_x_2 = target_theta_x
        target_theta_y_2 = target_theta_y
        target_theta_z_2 = target_theta_z


        rospy.loginfo(f"--- Moving to target Cartesian pose 2 ({target_x_2}, {target_y_2}, {target_z_2}) ---")
        success &= controller.move_to_cartesian_pose(target_x_2, target_y_2, target_z_2, target_theta_x_2, target_theta_y_2, target_theta_z_2)
        if not success: rospy.logwarn(f"Move to pose 2 failed or was aborted.")
        rospy.sleep(1)

        # 5. Go to a "home" or "retract" position
        # rospy.loginfo("--- Returning to a predefined home/safe pose ---")
        # success &= controller.example_home_the_robot() # Using our custom home/safe pose
        # if not success: rospy.logwarn("Moving to home/safe pose failed or was aborted.")

    if not success:
        rospy.logerr("The example encountered an error.")
    else:
        rospy.loginfo("Example completed successfully.")

if __name__ == "__main__":
    arm = SimplifiedArmController()
    arm.example_home_the_robot()
