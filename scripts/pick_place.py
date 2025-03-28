#!/usr/bin/env python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed 
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import rospy
import time

from kortex_driver.srv import *
from kortex_driver.msg import *


class ExampleFullArmMovement:
    def __init__(self):
        try:
            rospy.init_node('example_full_arm_movement_python')

            self.HOME_ACTION_IDENTIFIER = 2

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
        
            get_product_configuration_full_name = '/' + self.robot_name + '/base/get_product_configuration'
            rospy.wait_for_service(get_product_configuration_full_name)
            self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name, GetProductConfiguration)

            validate_waypoint_list_full_name = '/' + self.robot_name + '/base/validate_waypoint_list'
            rospy.wait_for_service(validate_waypoint_list_full_name)
            self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    
    def FillCartesianWaypoint(self, new_x, new_y, new_z, new_theta_x, new_theta_y, new_theta_z, blending_radius):
        waypoint = Waypoint()
        cartesianWaypoint = CartesianWaypoint()

        cartesianWaypoint.pose.x = new_x
        cartesianWaypoint.pose.y = new_y
        cartesianWaypoint.pose.z = new_z
        cartesianWaypoint.pose.theta_x = new_theta_x
        cartesianWaypoint.pose.theta_y = new_theta_y
        cartesianWaypoint.pose.theta_z = new_theta_z
        cartesianWaypoint.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        cartesianWaypoint.blending_radius = blending_radius
        waypoint.oneof_type_of_waypoint.cartesian_waypoint.append(cartesianWaypoint)

        return waypoint

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)

    def example_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def example_home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        self.last_action_notif_type = None
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()

    def example_set_cartesian_reference_frame(self):
        self.last_action_notif_type = None
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")

        # Wait a bit
        rospy.sleep(0.25)
        return True

    def example_send_cartesian_pose(self, reletive_pos=[0,0,0]):
        self.last_action_notif_type = None
        # Get the actual cartesian pose to increment it
        # You can create a subscriber to listen to the base_feedback
        # Here we only need the latest message in the topic though
        feedback = rospy.wait_for_message("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback)

        # Possible to execute waypointList via execute_action service or use execute_waypoint_trajectory service directly
        req = ExecuteActionRequest()
        trajectory = WaypointList()

        trajectory.waypoints.append(
            self.FillCartesianWaypoint(
                feedback.base.commanded_tool_pose_x+reletive_pos[0],
                feedback.base.commanded_tool_pose_y+reletive_pos[1],
                feedback.base.commanded_tool_pose_z+reletive_pos[2],
                feedback.base.commanded_tool_pose_theta_x,
                feedback.base.commanded_tool_pose_theta_y,
                feedback.base.commanded_tool_pose_theta_z,
                0)
        )

        trajectory.duration = 0
        trajectory.use_optimal_blending = False

        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)

        # Call the service
        rospy.loginfo("Sending the robot to the cartesian pose...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointTrajectory")
            return False
        else:
            return self.wait_for_action_end_or_abort()

    def example_send_joint_angles(self, joint_angles = [0,0,0,0,0,0]):
        self.last_action_notif_type = None

        req = ExecuteActionRequest()

        trajectory = WaypointList()
        waypoint = Waypoint()
        angularWaypoint = AngularWaypoint()

        # Angles to send the arm to vertical position (all zeros)
        for i in range(joint_angles):
            angularWaypoint.angles.append(joint_angles[i])

        # Each AngularWaypoint needs a duration and the global duration (from WaypointList) is disregarded. 
        # If you put something too small (for either global duration or AngularWaypoint duration), the trajectory will be rejected.
        angular_duration = 0
        angularWaypoint.duration = angular_duration

        # Initialize Waypoint and WaypointList
        waypoint.oneof_type_of_waypoint.angular_waypoint.append(angularWaypoint)
        trajectory.duration = 0
        trajectory.use_optimal_blending = False
        trajectory.waypoints.append(waypoint)

        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 30

        while (error_number >= 1 and angular_duration != MAX_ANGULAR_DURATION) :
            angular_duration += 1
            trajectory.waypoints[0].oneof_type_of_waypoint.angular_waypoint[0].duration = angular_duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(res.output.trajectory_error_report.trajectory_error_elements)

        if (angular_duration == MAX_ANGULAR_DURATION) :
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False

        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)
        
        # Send the angles
        rospy.loginfo("Sending the robot vertical...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointjectory")
            return False
        else:
            return self.wait_for_action_end_or_abort()

    def example_send_gripper_command(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True

    def example_cartesian_waypoint_action(self):
        self.last_action_notif_type = None

        req = ExecuteActionRequest()
        trajectory = WaypointList()

        config = self.get_product_configuration()

        if config.output.model == ModelId.MODEL_ID_L31:
            rospy.logerr("1")
            trajectory.waypoints.append(self.FillCartesianWaypoint(0.439,  0.194,  0.448, 90.6, -1.0, 150, 0))
            trajectory.waypoints.append(self.FillCartesianWaypoint(0.200,  0.150,  0.400, 90.6, -1.0, 150, 0))
            trajectory.waypoints.append(self.FillCartesianWaypoint(0.3,  0.050,  0.300, 90.6, -1.0, 150, 0))
            # trajectory.waypoints.append(self.FillCartesianWaypoint(0, 0, 0, 0, 0, 0,0))
            # trajectory.waypoints.append(self.FillCartesianWaypoint(0.3, -0.519, 1.572, -89, -46, 0.3,0))
            # trajectory.waypoints.append(self.FillCartesianWaypoint(0.4, -0.722, 1.572, -89, -46, 0.3,0))

        
        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)
        
        # Call the service
        rospy.loginfo("Executing Kortex action ExecuteWaypointTrajectory...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call action ExecuteWaypointTrajectory")
            return False
        else:
            return self.wait_for_action_end_or_abort()
        
    def pick_place(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/full_arm_movement_python")
        except:
            pass

        if success:
            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.example_clear_faults()
            #*******************************************************************************
            
            #*******************************************************************************
            # Activate the action notifications
            success &= self.example_subscribe_to_a_robot_notification()
            #*******************************************************************************

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's fully open the gripper
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0)
            else:
                rospy.logwarn("No gripper is present on the arm.")
            # 到达抓取位置
            if success:
                success &= self.example_send_joint_angles([1.2, -0.722, 1.572, -1.566, -0.807, 0.3])
            else:
                rospy.logerr("Fail to reach pick position")
            # 下降高度
            if success:
                success &= self.example_send_cartesian_pose([0,0,-0.06])
            else:
                rospy.logerr("Fail to move downward")
            # 抓取
            if success:
                success &= self.example_send_gripper_command(0.9)
            else:
                rospy.logerr("Fail to pick up")
            # 到达中途位置
            if success:
                success &= self.example_send_joint_angles([0.9, -0.519, 1.572, -1.566, -1.028, 0.3])
            else:
                rospy.logerr("Fail to move to mid waypoint")
            # 到达释放位置
            if success:
                success &= self.example_send_joint_angles([0, -0.722, 1.572, -1.566, -0.807, 0.3])
            else:
                rospy.logerr("Fail to move to the place position")
            # 下降高度
            if success:
                success &= self.example_send_cartesian_pose([0,0,-0.06])
            else:
                rospy.logerr("Fail to move downward")
            # 释放
            if success:
                success &= self.example_send_gripper_command(0)
            else:
                rospy.logerr("Fail to place")
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's close the gripper at 50%
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0.5)
            else:
                rospy.logwarn("No gripper is present on the arm.")    
            #*******************************************************************************
        
            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of waypoint
            # Let's move the arm
            success &= self.example_cartesian_waypoint_action()

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/full_arm_movement_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")
      

    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/full_arm_movement_python")
        except:
            pass

        if success:
            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.example_clear_faults()
            #*******************************************************************************
            
            #*******************************************************************************
            # Activate the action notifications
            success &= self.example_subscribe_to_a_robot_notification()
            #*******************************************************************************

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's fully open the gripper
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0.0)
            else:
                rospy.logwarn("No gripper is present on the arm.")  
            #*******************************************************************************

            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.example_set_cartesian_reference_frame()

            # Example of cartesian pose
            # Let's make it move in Z
            success &= self.example_send_cartesian_pose()
            #*******************************************************************************

            #*******************************************************************************
            # Example of angular position
            # Let's send the arm to vertical position
            success &= self.example_send_joint_angles()
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's close the gripper at 50%
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0.5)
            else:
                rospy.logwarn("No gripper is present on the arm.")    
            #*******************************************************************************
        
            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of waypoint
            # Let's move the arm
            success &= self.example_cartesian_waypoint_action()

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/full_arm_movement_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")



# def gripper_test():
#   example = ExampleMoveItTrajectories()

#   # For testing purposes
#   success = example.is_init_success

#   try:
#       rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
#   except:
#       pass
  
#   if example.is_gripper_present and success:
#     rospy.loginfo("Reaching Named Target Home")
#     success &= example.reach_gripper_position(1)
#     time.sleep(1)
#     success &= example.reach_gripper_position(0)
#     print(success)
#   else:
#     rospy.logerr("Move to home failed -- 1")


# def pick_place():
#   example = ExampleMoveItTrajectories()

#   # For testing purposes
#   success = example.is_init_success

#   try:
#       rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
#   except:
#       pass
  
#   if success:
#     rospy.loginfo("Reaching Named Target Home")
#     success &= example.reach_named_position("home")
#     print(success)
#   else:
#     rospy.logerr("Move to home failed -- 1")


#   if success:
#     # success &= example.reach_gripper_position(1)
#     print("success")
#   else:
#     rospy.logerr("Open grip failed -- 2")


#   if success:
#     rospy.loginfo("Move to target orientation pick 1")
#     ang = [1.2, -0.722, 1.572, -1.566, -0.807, 0.3]
#     success &= example.reach_joint_angles(tolerance=0.01,angle=ang)
#     print(success)
#   else:
#     rospy.logerr("Move to pick position failed -- 3")


#   if success:
#     rospy.loginfo("Reaching Cartesian Pose...")
#     actual_pose = example.get_cartesian_pose()
#     rospy.logwarn(actual_pose)
#     actual_pose.position.z -= 0.06
#     success &= example.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
#     print(success)
#   else:
#     rospy.logerr("Move downward failed -- 4")

    
#   # if example.is_gripper_present and success:
#   #   rospy.loginfo("Opening the gripper... pick ")
#   #   success &= example.reach_gripper_position(0)
#   #   print (success)

      
#   if success:
#     rospy.loginfo("Closing the gripper... pick ")
#     # success &= example.reach_gripper_position(0.2)
#     print (success)
#   else:
#     rospy.logerr("Gripper close failed -- 5")

#   #   rospy.loginfo("Closing the gripper 50%...")
#   #   success &= example.reach_gripper_position(0.5)
#   #   print (success)

#   if success:
#     rospy.loginfo("Move to target orientation place 1")
#     ang = [0.9, -0.519, 1.572, -1.566, -1.028, 0.3]
#     success &= example.reach_joint_angles(tolerance=0.01,angle=ang)
#     print(success)
#   else:
#     rospy.logerr("Move to mid-pos failed -- 6")

#   if success:
#     rospy.loginfo("Move to target orientation place 2")
#     ang = [0, -0.722, 1.572, -1.566, -0.807, 0.3]
#     success &= example.reach_joint_angles(tolerance=0.01,angle=ang)
#     print(success)
#   else:
#     rospy.logerr('Move to final pos failed -- 7')


#   if success:
#     rospy.loginfo("Reaching Cartesian Pose...")
#     actual_pose = example.get_cartesian_pose()
#     rospy.logwarn(actual_pose)
#     actual_pose.position.z -= 0.06
#     success &= example.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
#     print(success)
#   else:
#     rospy.logerr("Move downward failed -- 4")

    
#   if success:
#     rospy.loginfo("Opening the gripper... place")
#     # success &= example.reach_gripper_position(1)
#     print (success)
#   else:
#     rospy.logerr('open gripper failed -- 8')


#   # For testing purposes
#   rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

#   if not success:
#       rospy.logerr("The example encountered an error.")

  



if __name__ == "__main__":
    ex = ExampleFullArmMovement()
    ex.pick_place()