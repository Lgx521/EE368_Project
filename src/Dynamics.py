import math

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

class Link:
    def __init__(self, dh_params):
        self.dh_params_ = dh_params

    def transformation_matrix(self, theta):
        alpha = self.dh_params_[0]
        a = self.dh_params_[1]
        d = self.dh_params_[2]
        theta = theta+self.dh_params_[3]
        st = math.sin(theta)
        ct = math.cos(theta)
        sa = math.sin(alpha)
        ca = math.cos(alpha)
        trans = np.array([[ct, -st, 0, a],
                          [st*ca, ct * ca, - sa, -sa * d],
                          [st*sa, ct * sa,   ca,  ca * d],
                          [0, 0, 0, 1]])
        return trans

    def set_inertial_parameters(self, mass, center: list, inertia, T_dh_link):
        self.mass = mass
        ixx = inertia[0]
        ixy = inertia[1]
        ixz = inertia[2]
        iyy = inertia[3]
        iyz = inertia[4]
        izz = inertia[5]
        I = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        R = T_dh_link[0:3, 0:3]
        new_I = R.dot(I).dot(R.T)
        center.append(1.0)
        new_center = T_dh_link.dot(np.array(center).T)

        self.center = new_center[:3]
        self.inertia_tensor = new_I
        print(f"center of mass: {self.center}")
        print(f"inertia tensor: {self.inertia_tensor}")

    @staticmethod
    def basic_jacobian(trans, ee_pos):
        pos = np.array(
            [trans[0, 3], trans[1, 3], trans[2, 3]])
        z_axis = np.array(
            [trans[0, 2], trans[1, 2], trans[2, 2]])

        basic_jacobian = np.hstack(
            (np.cross(z_axis, ee_pos - pos), z_axis))
        return basic_jacobian


class NLinkArm:

    def __init__(self, dh_params_list) -> None:
        self.link_list = []
        for i in range(len(dh_params_list)):
            self.link_list.append(Link(dh_params_list[i]))

    def transformation_matrix(self, thetas):
        trans = np.identity(4)
        for i in range(len(self.link_list)):
            trans = np.dot(
                trans, self.link_list[i].transformation_matrix(thetas[i]))
        return trans

    def forward_kinematics(self, thetas):
        trans = self.transformation_matrix(thetas)
        x = trans[0, 3]
        y = trans[1, 3]
        z = trans[2, 3]

        alpha, beta, gamma = self.euler_angle(thetas)
        return [x, y, z, alpha, beta, gamma]

    def euler_angle(self, thetas):
        trans = self.transformation_matrix(thetas)

        alpha = math.atan2(trans[1][2], trans[0][2])
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) + math.pi
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) - math.pi
        beta = math.atan2(
            trans[0][2] * math.cos(alpha) + trans[1][2] * math.sin(alpha),
            trans[2][2])
        gamma = math.atan2(
            -trans[0][0] * math.sin(alpha) + trans[1][0] * math.cos(alpha),
            -trans[0][1] * math.sin(alpha) + trans[1][1] * math.cos(alpha))

        return alpha, beta, gamma

    def inverse_kinematics(self, ref_ee_pose):
        thetas = [0, 0, 0, 0, 0, 0]
        for cnt in range(5000):
            ee_pose = self.forward_kinematics(thetas)
            diff_pose = np.array(ref_ee_pose) - ee_pose

            basic_jacobian_mat = self.basic_jacobian(thetas)
            alpha, beta, gamma = self.euler_angle(thetas)

            K_zyz = np.array(
                [[0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
                 [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
                 [1, 0, math.cos(beta)]])
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz

            theta_dot = np.dot(
                np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
                np.array(diff_pose))
            thetas = thetas + theta_dot / 100.
            if np.linalg.norm(theta_dot) < 0.001:
                break
        # thetas = np.mod(thetas, 2*np.pi)
        return thetas

    def basic_jacobian(self, thetas):
        ee_pos = self.forward_kinematics(thetas)[0:3]
        basic_jacobian_mat = []
        trans = np.identity(4)
        for i in range(len(self.link_list)):
            trans = np.dot(
                trans, self.link_list[i].transformation_matrix(thetas[i]))
            basic_jacobian_mat.append(
                self.link_list[i].basic_jacobian(trans, ee_pos))
        return np.array(basic_jacobian_mat).T

    def get_torque(self, thetas, thetas_d, theta_dd, f_ext, n_ext):
        f_ext = np.array(f_ext).T
        n_ext = np.array(n_ext).T

        link_num = len(self.link_list)
        R_i_iplus1_list = np.zeros((3,3,link_num))
        P_i_iplus1_list = np.zeros((3,link_num))
        P_i_c_list = np.zeros((3,link_num+1))
        for i in range(link_num):
            T_i_iplus1 = self.link_list[i].transformation_matrix(thetas[i])
            R_i_iplus1_list[:,:,i] = T_i_iplus1[:3,:3]
            P_i_iplus1_list[:,i] = T_i_iplus1[:3, 3]
            P_i_c_list[:,i+1] = self.link_list[i].center

        omega = np.zeros((3, link_num+1))
        omega_d = np.zeros((3, link_num+1))
        v_dot_i = np.zeros((3, link_num+1))
        v_dot_c = np.zeros((3, link_num+1))
        v_dot_i[:, 0] = [0, 0, 9.8]
        F = np.zeros((3, link_num+1))
        N = np.zeros((3, link_num+1))

        for i in range(link_num):
            R = R_i_iplus1_list[:,:,i].T
            m = self.link_list[i].mass
            P_i_iplus1 = P_i_iplus1_list[:,i]
            P_iplus1_c = P_i_c_list[:,i+1]
            I_iplus1 = self.link_list[i].inertia_tensor
            theta_dot_z = thetas_d[i]*np.array([0, 0, 1]).T
            omega[:, i+1] = R.dot(omega[:, i]) + theta_dot_z
            omega_d[:, i+1] = R.dot(omega_d[:, i]) + np.cross(
                R.dot(omega[:, i]), theta_dot_z) + theta_dd[i]*np.array([0, 0, 1]).T
            v_dot_i[:, i+1] = R.dot(np.cross(omega_d[:, i], P_i_iplus1)+np.cross(
                omega_d[:, i], np.cross(omega_d[:, i], P_i_iplus1))+v_dot_i[:, i])
            v_dot_c[:, i+1] = np.cross(omega_d[:, i+1], P_iplus1_c) + np.cross(
                omega[:, i+1], np.cross(omega[:, i+1], P_iplus1_c)) + v_dot_i[:, i+1]
            F[:, i+1] = m*v_dot_c[:, i+1]
            N[:, i+1] = I_iplus1.dot(omega_d[:, i+1]) + \
                np.cross(omega[:, i+1], I_iplus1.dot(omega[:, i+1]))

        f = np.zeros((3, link_num+1))
        n = np.zeros((3, link_num+1))
        tau = np.zeros(link_num+1)

        for i in range(link_num, 0, -1):
            R = T_i_iplus1[:3, :3]
            if i == link_num:
                f[:,i] = f_ext + F[:,i]
                n[:,i] = N[:,i] + n_ext + np.cross(P_i_c_list[:,i],F[:,i])
                tau[i] = n[:,i].T.dot(np.array([0, 0, 1]).T)
            else:
                R = R_i_iplus1_list[:,:,i]
                f[:,i] = R.dot(f[:,i+1]) + F[:,i]
                n[:,i] = N[:,i] + R.dot(n[:,i+1]) + np.cross(P_i_c_list[:,i],F[:,i]) + np.cross(P_i_iplus1_list[:,i],R.dot(f[:,i+1]))
                tau[i] = n[:,i].T.dot(np.array([0, 0, 1]).T)
        return tau[1:]


if __name__ == "__main__":
    rospy.init_node("dynamics_test")
    real_torque_pub_list = []
    sim_torque_pub_list = []

    for i in range(6):
        real_pub = rospy.Publisher(f"/real_torques/joint_{i}",Float64,queue_size=1)
        real_torque_pub_list.append(real_pub)
        sim_pub = rospy.Publisher(f"/sim_torques/joint_{i}",Float64,queue_size=1)
        sim_torque_pub_list.append(sim_pub)

    dh_params_list = np.array([[0, 0, 243.25/1000, 0],
                               [math.pi/2, 0, 30/1000, 0+math.pi/2],
                               [math.pi, 280/1000, 20/1000, 0+math.pi/2],
                               [math.pi/2, 0, 245/1000, 0+math.pi/2],
                               [math.pi/2, 0, 57/1000, 0],
                               [-math.pi/2, 0, 235/1000, 0-math.pi/2]])
    gen3_lite = NLinkArm(dh_params_list)
    gen3_lite.link_list[0].set_inertial_parameters(0.95974404, [
                                                   2.477E-05, 0.02213531, 0.09937686], [0.00165947, 2e-08, 3.6E-07, 0.00140355, 0.00034927, 0.00089493],
                                                   np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -115/1000], [0, 0, 0, 1]]))
    gen3_lite.link_list[1].set_inertial_parameters(1.17756164, [0.02998299, 0.21154808, 0.0453031], [
                                                   0.01149277, 1E-06, 1.6E-07, 0.00102851, 0.00140765, 0.01133492],
                                                   np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    gen3_lite.link_list[2].set_inertial_parameters(0.59767669, [0.0301559, 0.09502206, 0.0073555], [
                                                   0.00163256, 7.11E-06, 1.54E-06, 0.00029798, 9.587E-05, 0.00169091],
                                                   np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -20/1000], [0, 0, 0, 1]]))
    gen3_lite.link_list[3].set_inertial_parameters(0.52693412, [0.00575149, 0.01000443, 0.08719207], [
                                                   0.00069098, 2.4E-07, 0.00016483, 0.00078519, 7.4E-07, 0.00034115],
                                                   np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, -105/1000], [0, 0, 0, 1]]))
    gen3_lite.link_list[4].set_inertial_parameters(0.58097325, [0.08056517, 0.00980409, 0.01872799], [
                                                   0.00021268, 5.21E-06, 2.91E-06, 0.00106371, 1.1E-07, 0.00108465],
                                                   np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, -28.5/1000], [0, 0, 0, 1]]))
    gen3_lite.link_list[5].set_inertial_parameters(0.2018, [0.00993, 0.00995, 0.06136], [
                                                   0.0003428, 0.00000019, 0.0000001, 0.00028915, 0.00000027, 0.00013076],
                                                   np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -130/1000], [0, 0, 0, 1]]))

    last_velocities = [0,0,0,0,0,0]
    last_time = rospy.get_time()
    while not rospy.is_shutdown():
        feedback = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState)
        thetas = feedback.position[0:6]
        velocities = feedback.velocity[0:6]
        torques = np.array(feedback.effort[0:6])
        thetas_d = velocities
        dt = rospy.get_time() - last_time
        last_time = rospy.get_time()
        thetas_dd = np.subtract(velocities,last_velocities)/dt
        last_velocities = velocities
        sim_torque = gen3_lite.get_torque(thetas,thetas_d,thetas_dd,[0,0,0],[0,0,0])
        for i in range(6):
            real_torque_pub_list[i].publish(torques[i])
            sim_torque_pub_list[i].publish(sim_torque[i])
        print(f"joint torque: {torques}")
        print(f"sim torque: {sim_torque}")
        print(f"diff {np.subtract(torques,sim_torque)}")
