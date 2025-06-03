#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import your IK solver and DH parameters
try:
    import inverse_kinematics as ik
except ImportError:
    rospy.logerr("Failed to import inverse_kinematics.py. Make sure it's in the same directory or PYTHONPATH.")
    exit(1)

from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

class KinovaCubicTrajectoryExecutor:
    def __init__(self):
        rospy.init_node('kinova_cubic_trajectory_executor')

        # --- Parameters ---
        # Action server name for FollowJointTrajectory
        self.arm_joint_trajectory_action_name = rospy.get_param(
            "~arm_joint_trajectory_action_name",
            "/my_gen3/gen3_joint_trajectory_controller/follow_joint_trajectory" # MODIFY THIS if different
        )
        # Topic for current joint states
        self.joint_states_topic = rospy.get_param(
            "~joint_states_topic",
            "/my_gen3/joint_states" # MODIFY THIS if different (e.g., /my_gen3/base_feedback for Kortex API direct messages)
        )
        # Number of joints (should match your DH parameters)
        self.num_joints = ik.num_joints
        # Joint names (MUST MATCH the order in DH params and what the controller expects)
        self.joint_names = [f"joint_{i+1}" for i in range(self.num_joints)]
        # Example: For Gen3, it might be specific like:
        # self.joint_names = rospy.get_param("~joint_names", ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])


        rospy.loginfo(f"Expecting {self.num_joints} joints with names: {self.joint_names}")
        rospy.loginfo(f"Using FollowJointTrajectory action server: {self.arm_joint_trajectory_action_name}")
        rospy.loginfo(f"Subscribing to joint states on: {self.joint_states_topic}")

        # --- ROS Subscribers and Action Clients ---
        self.current_joint_angles = None
        self.joint_state_received = False
        rospy.Subscriber(self.joint_states_topic, JointState, self.joint_states_callback)

        self.trajectory_client = actionlib.SimpleActionClient(
            self.arm_joint_trajectory_action_name,
            FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for FollowJointTrajectory action server...")
        if not self.trajectory_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr(f"Action server {self.arm_joint_trajectory_action_name} not available after 10s. Exiting.")
            rospy.signal_shutdown("Action server not available")
            exit(1)
        rospy.loginfo("Connected to FollowJointTrajectory action server.")

        # --- DH Parameters from your IK script ---
        self.dh_params = ik.dh_parameters

        # --- Joint Limits (Optional but Recommended for IK) ---
        # Example: Kinova Gen3 (6DOF) approx limits in radians
        # You should get the precise limits for your specific robot model
        # These are used by the IK solver.
        # Format: [[min1, max1], [min2, max2], ...]
        self.joint_limits_rad = rospy.get_param("~joint_limits_rad", None)
        if self.joint_limits_rad:
            rospy.loginfo(f"Using joint limits for IK: {self.joint_limits_rad}")
            if len(self.joint_limits_rad) != self.num_joints:
                rospy.logwarn("Mismatch between num_joints and provided joint_limits_rad length. Ignoring limits.")
                self.joint_limits_rad = None
        else:
            rospy.loginfo("No joint limits provided for IK. IK might produce out-of-range solutions.")
            # You might want to hardcode default approximate limits if not provided
            # self.joint_limits_rad = [
            #     [-np.deg2rad(360), np.deg2rad(360)], # Joint 1 (example, Kinova is continuous or large range)
            #     [-np.deg2rad(128.8), np.deg2rad(128.8)], # Joint 2
            #     [-np.deg2rad(360), np.deg2rad(360)], # Joint 3
            #     [-np.deg2rad(147.8), np.deg2rad(147.8)], # Joint 4
            #     [-np.deg2rad(360), np.deg2rad(360)], # Joint 5
            #     [-np.deg2rad(120.3), np.deg2rad(120.3)]  # Joint 6
            # ]
            # rospy.loginfo(f"Using default approximate joint limits for IK: {self.joint_limits_rad}")


    def joint_states_callback(self, msg):
        """
        Callback for joint state updates.
        Stores current joint angles in the correct order.
        """
        if not self.joint_state_received:
            rospy.loginfo("First JointState message received.")

        # Ensure we get angles for all named joints and in the correct order
        current_angles_dict = dict(zip(msg.name, msg.position))
        ordered_angles = []
        try:
            for name in self.joint_names:
                ordered_angles.append(current_angles_dict[name])
            self.current_joint_angles = np.array(ordered_angles)
            self.joint_state_received = True
        except KeyError as e:
            if not self.joint_state_received: # Only log error if we haven't successfully received yet
                rospy.logwarn_throttle(5.0, f"Joint '{e}' not found in JointState message. Waiting... Message names: {msg.name}")


    def get_current_joint_angles(self, timeout_sec=5.0):
        """
        Waits for and returns the current joint angles.
        """
        start_time = rospy.Time.now()
        while not self.joint_state_received and (rospy.Time.now() - start_time).to_sec() < timeout_sec:
            rospy.sleep(0.1)
        if not self.joint_state_received:
            rospy.logerr("Failed to receive joint states.")
            return None
        return self.current_joint_angles

    def _cubic_coeffs(self, q0, qf, v0, vf, T):
        """
        Calculates coefficients for a cubic polynomial:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        Given q(0)=q0, q(T)=qf, q_dot(0)=v0, q_dot(T)=vf
        """
        if T <= 0:
            # Return static trajectory if duration is zero or negative
            return np.array([q0, 0, 0, 0])

        a0 = q0
        a1 = v0
        a2 = (3*(qf - q0) - (2*v0 + vf)*T) / (T**2)
        a3 = (2*(q0 - qf) + (v0 + vf)*T) / (T**3)
        return np.array([a0, a1, a2, a3])

    def generate_cubic_trajectory(self, q_start, q_end, duration, num_points=100):
        """
        Generates a joint trajectory using cubic polynomial interpolation for each joint.
        Assumes zero start and end velocities for the trajectory segment.
        q_start: array of starting joint angles
        q_end: array of ending joint angles
        duration: total time for the trajectory in seconds
        num_points: number of points in the trajectory
        """
        if duration <= 0:
            rospy.logwarn("Trajectory duration must be positive. Setting to 0.1s")
            duration = 0.1

        trajectory_points = []
        time_step = duration / (num_points - 1) if num_points > 1 else duration

        # Coefficients for each joint (assuming zero start/end velocity for the segment)
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # q_dot(t) = a1 + 2*a2*t + 3*a3*t^2
        v_start = np.zeros_like(q_start)
        v_end = np.zeros_like(q_end)

        coeffs_per_joint = []
        for i in range(self.num_joints):
            coeffs = self._cubic_coeffs(q_start[i], q_end[i], v_start[i], v_end[i], duration)
            coeffs_per_joint.append(coeffs)
        coeffs_per_joint = np.array(coeffs_per_joint) # Shape: (num_joints, 4)

        for i in range(num_points):
            t = i * time_step
            point = JointTrajectoryPoint()
            point.positions = [0.0] * self.num_joints
            point.velocities = [0.0] * self.num_joints
            # Accelerations can optionally be set if your controller uses them
            # point.accelerations = [0.0] * self.num_joints
            point.time_from_start = rospy.Duration.from_sec(t)

            for j in range(self.num_joints):
                c = coeffs_per_joint[j] # a0, a1, a2, a3 for joint j
                point.positions[j] = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3
                point.velocities[j] = c[1] + 2*c[2]*t + 3*c[3]*t**2
            
            trajectory_points.append(point)
        
        # Ensure the last point exactly matches q_end and has zero velocity if num_points > 1
        if num_points > 1:
            trajectory_points[-1].positions = list(q_end)
            trajectory_points[-1].velocities = list(v_end) # Should be zero by cubic formulation for vf=0

        return trajectory_points


    def plan_and_execute_to_pose(self, target_position, target_orientation_euler, trajectory_duration=5.0):
        """
        Plans and executes a trajectory to a target Cartesian pose.
        target_position: [x, y, z]
        target_orientation_euler: [roll, pitch, yaw] in radians
        trajectory_duration: Time in seconds for the movement
        """
        rospy.loginfo("Getting current joint angles...")
        q_initial = self.get_current_joint_angles()
        if q_initial is None:
            rospy.logerr("Could not get current joint angles. Aborting.")
            return False
        rospy.loginfo(f"Current joint angles (rad): {np.round(q_initial,3)}")

        # 1. Convert target pose to 4x4 matrix
        target_rotation_matrix = R.from_euler('xyz', target_orientation_euler, degrees=False).as_matrix()
        T_target = np.eye(4)
        T_target[:3, :3] = target_rotation_matrix
        T_target[:3, 3] = target_position
        rospy.loginfo("Target Pose (Homogeneous Matrix):\n" + str(np.round(T_target, 4)))

        # 2. Solve Inverse Kinematics
        rospy.loginfo("Solving IK...")
        # Use current angles as initial guess for IK
        q_target_ik, converged, pos_err, ori_err = ik.inverse_kinematics_optimizer(
            T_target,
            q_initial, # Using current angles as initial guess
            self.dh_params,
            joint_limits=self.joint_limits_rad,
            pos_weight=10.0, # Higher weight for position
            ori_weight=1.0,
            pos_tolerance=1e-3, # Looser tolerance for faster planning for demo
            ori_tolerance=1e-2,
            max_iterations=200,
            optimizer_ftol=1e-6
        )

        if not converged:
            rospy.logerr(f"IK solver did not converge or solution error too high. Pos Err: {pos_err:.4e}, Ori Err: {ori_err:.4e}")
            rospy.logerr(f"Target q (not executed): {np.round(q_target_ik,3)}")
            # You might want to check FK of q_target_ik to see how far off it is
            T_final_attempt, _ = ik.forward_kinematics(q_target_ik, self.dh_params)
            rospy.logerr("FK of IK attempt:\n" + str(np.round(T_final_attempt, 4)))
            return False

        rospy.loginfo(f"IK Solution (rad): {np.round(q_target_ik,3)}")
        rospy.loginfo(f"IK Pos Err: {pos_err:.4e}, Ori Err: {ori_err:.4e}")

        # Check if the solution is reasonably different from the start
        if np.linalg.norm(np.array(q_initial) - np.array(q_target_ik)) < 1e-3: # Threshold for "no change"
            rospy.loginfo("Target joint angles are very close to current ones. No motion needed.")
            return True

        # 3. Generate Cubic Trajectory
        rospy.loginfo(f"Generating cubic trajectory from current to target joint angles over {trajectory_duration}s...")
        trajectory_points = self.generate_cubic_trajectory(
            q_initial,
            q_target_ik,
            trajectory_duration,
            num_points=100  # Adjust number of points for smoothness vs. message size
        )

        if not trajectory_points:
            rospy.logerr("Failed to generate trajectory.")
            return False

        # 4. Create FollowJointTrajectoryGoal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = trajectory_points
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.2) # Start shortly after sending

        # Path and goal tolerances (optional, but good practice)
        # If the controller deviates too much, it might abort
        # Set to zero or empty list [] to use controller defaults
        # goal.path_tolerance = []
        # goal.goal_tolerance = []
        # goal.goal_time_tolerance = rospy.Duration(0.5)


        # 5. Send Goal to Action Server
        rospy.loginfo("Sending trajectory goal to action server...")
        self.trajectory_client.send_goal(goal)

        rospy.loginfo("Waiting for trajectory execution...")
        # Wait for the server to finish performing the action.
        # You can add a timeout here if needed.
        # Example: self.trajectory_client.wait_for_result(rospy.Duration(trajectory_duration + 5.0))
        self.trajectory_client.wait_for_result()

        result_status = self.trajectory_client.get_state()
        result = self.trajectory_client.get_result()

        if result_status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Trajectory executed successfully!")
            return True
        else:
            rospy.logerr(f"Trajectory execution failed. Status: {result_status}")
            if result and result.error_code != 0 : # SUCCESS = 0
                 rospy.logerr(f"Error code: {result.error_code}, Error string: {result.error_string}")
            return False

if __name__ == "__main__":
    try:
        executor = KinovaCubicTrajectoryExecutor()

        # --- Wait for first joint state message ---
        rospy.loginfo("Waiting for initial joint states...")
        q_start_verify = executor.get_current_joint_angles(timeout_sec=10.0)
        if q_start_verify is None:
            rospy.logerr("Could not get initial joint states. Shutting down.")
            rospy.signal_shutdown("Failed to get initial joint states")
            exit(1)
        rospy.loginfo(f"Initial joint states received: {np.round(q_start_verify,3)}")

        # --- Define Target Pose 1 ---
        # MODIFY THESE VALUES FOR YOUR DESIRED TARGET
        target_pos1 = np.array([0.4, 0.1, 0.3])      # meters [x, y, z]
        target_ori_euler1 = np.array([np.pi, 0, np.pi/4]) # radians [roll, pitch, yaw]
        # Note: Euler angles can have singularities. Quaternions are often preferred for targets.
        # Your IK uses matrices, so this is fine.

        rospy.loginfo("\n" + "="*30 + "\nMOVING TO TARGET 1\n" + "="*30)
        if executor.plan_and_execute_to_pose(target_pos1, target_ori_euler1, trajectory_duration=7.0):
            rospy.loginfo("Successfully moved to Target 1.")
        else:
            rospy.logerr("Failed to move to Target 1.")

        rospy.sleep(1.0) # Pause between movements

        # --- Define Target Pose 2 (e.g., a different orientation or position) ---
        target_pos2 = np.array([0.3, -0.2, 0.4])
        target_ori_euler2 = np.array([np.pi*0.8, np.pi*0.1, -np.pi/3])

        rospy.loginfo("\n" + "="*30 + "\nMOVING TO TARGET 2\n" + "="*30)
        if executor.plan_and_execute_to_pose(target_pos2, target_ori_euler2, trajectory_duration=8.0):
            rospy.loginfo("Successfully moved to Target 2.")
        else:
            rospy.logerr("Failed to move to Target 2.")
        
        rospy.loginfo("\n" + "="*30 + "\nGOING BACK TO APPROX HOME (using FK of all zeros for target)\n" + "="*30)
        # Example: Go to a "home" defined by joint angles [0,0,0,0,0,0]
        # First, find the Cartesian pose of this home configuration
        q_home_joints = np.zeros(executor.num_joints) # All joints at zero
        T_home_cartesian, _ = ik.forward_kinematics(q_home_joints, executor.dh_params)
        home_pos = T_home_cartesian[:3,3]
        home_rot_mat = T_home_cartesian[:3,:3]
        home_ori_euler = R.from_matrix(home_rot_mat).as_euler('xyz', degrees=False)

        if executor.plan_and_execute_to_pose(home_pos, home_ori_euler, trajectory_duration=7.0):
            rospy.loginfo("Successfully moved to approximate home pose.")
        else:
            rospy.logerr("Failed to move to approximate home pose.")


        rospy.loginfo("Script finished.")

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()