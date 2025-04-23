# 导入必要的库
import numpy as np # 用于数值计算，特别是矩阵和向量操作
import rospy # ROS Python 客户端库，用于与 ROS 系统交互
from sensor_msgs.msg import JointState # ROS 标准消息类型，用于表示机器人的关节状态
from geometry_msgs.msg import Point # ROS 标准消息类型，用于表示三维空间中的点（这里用来发布位置、速度、力）
import math

# 定义 Link 类，代表机械臂的单个连杆
class Link:
    # 初始化函数，接收一个连杆的 DH 参数列表
    def __init__(self, dh_params):
        """
        构造函数，初始化 Link 对象。
        :param dh_params: 包含单个连杆 DH 参数的列表或元组，顺序通常为 [alpha, a, d, theta_offset]
                          alpha: 扭转角 (twist angle) - 绕新 X 轴旋转
                          a: 连杆长度 (link length) - 沿新 X 轴平移
                          d: 连杆偏移 (link offset) - 沿旧 Z 轴平移
                          theta_offset: 关节角偏移 (joint angle offset) - 初始的 theta 角，实际 theta 会加上这个偏移
        """
        self.dh_params_ = dh_params # 将传入的 DH 参数存储为实例变量

    # 计算从当前连杆坐标系到下一个连杆坐标系的变换矩阵
    def transformation_matrix(self, theta):
        """
        根据给定的关节变量 theta 计算此连杆的齐次变换矩阵。
        使用标准的 Denavit-Hartenberg (DH) 约定。
        :param theta: 当前关节的角度变量 (rad)
        :return: 4x4 的齐次变换矩阵 (NumPy array)
        """
        # 从存储的 DH 参数中提取 alpha, a, d
        alpha = self.dh_params_[0]
        a = self.dh_params_[1]
        d = self.dh_params_[2]
        # 计算实际的 theta 角，将输入的关节变量加上固定的偏移量
        theta = theta + self.dh_params_[3]

        # 预计算三角函数值，提高效率
        st = np.sin(theta)
        ct = np.cos(theta)
        sa = np.sin(alpha)
        ca = np.cos(alpha)

        # 构建标准的 DH 变换矩阵
        # T = Rot(Z, theta) * Trans(Z, d) * Trans(X, a) * Rot(X, alpha)
        trans = np.array([[ct, -st, 0, a],
                          [st*ca, ct * ca, -sa, -sa * d],
                          [st*sa, ct * sa,   ca,  ca * d],
                          [0, 0, 0, 1]])
        return trans # 返回计算得到的 4x4 变换矩阵

    @staticmethod # 声明这是一个静态方法，可以不通过实例直接调用 Link.basic_jacobian(...)
    def basic_jacobian(trans, ee_pos):
        """
        计算单个旋转关节对末端执行器速度贡献的雅可比矩阵列（基本雅可比）。
        这个方法计算的是几何雅可比矩阵的一列。
        适用于旋转关节 (Revolute Joint)。
        :param trans: 从基坐标系到当前关节坐标系的累积变换矩阵 (4x4 NumPy array)
        :param ee_pos: 末端执行器在基坐标系中的位置 (3x1 NumPy array or list/tuple)
        :return: 6x1 的雅可比矩阵列 (NumPy array)，前三行是线性速度贡献，后三行是角速度贡献。
        """
        # 从变换矩阵中提取当前关节坐标系的原点在基坐标系中的位置
        pos = np.array(
            [trans[0, 3], trans[1, 3], trans[2, 3]])
        # 从变换矩阵中提取当前关节坐标系的 Z 轴在基坐标系中的方向向量
        z_axis = np.array(
            [trans[0, 2], trans[1, 2], trans[2, 2]])

        # 计算雅可比矩阵列
        # 线性速度部分：v = omega x r = z_axis x (ee_pos - pos)
        # 角速度部分：omega = z_axis (对于绕 Z 轴旋转的关节)
        basic_jacobian_col = np.hstack( # 水平堆叠两个数组
            (np.cross(z_axis, ee_pos - pos), z_axis)) # cross 是叉乘运算
        return basic_jacobian_col # 返回 6 维向量


# 定义 NLinkArm 类，代表整个 N 连杆机械臂
class NLinkArm:
    # 初始化函数，接收一个包含所有连杆 DH 参数的列表
    def __init__(self, dh_params_list) -> None:
        """
        构造函数，初始化 NLinkArm 对象。
        :param dh_params_list: 一个列表，其中每个元素是对应连杆的 DH 参数列表 (如 [[alpha1, a1, d1, theta_offset1], [alpha2, a2, d2, theta_offset2], ...])
        """
        self.link_list = [] # 初始化一个空列表，用于存储 Link 对象
        # 遍历传入的 DH 参数列表
        for i in range(len(dh_params_list)):
            # 为每个连杆的 DH 参数创建一个 Link 对象，并添加到 link_list 中
            self.link_list.append(Link(dh_params_list[i]))

    # 计算从基坐标系到末端执行器坐标系的总变换矩阵
    def transformation_matrix(self, thetas):
        """
        根据所有关节的角度计算从基座到末端执行器的总齐次变换矩阵。
        :param thetas: 包含所有关节角度的列表或数组 (rad)
        :return: 4x4 的总齐次变换矩阵 (NumPy array) T_0^N
        """
        # 初始化总变换矩阵为单位矩阵（表示基坐标系自身）
        trans = np.identity(4)
        # 遍历臂上的每一个连杆
        for i in range(len(self.link_list)):
            # 获取当前连杆相对于前一连杆的变换矩阵
            link_trans = self.link_list[i].transformation_matrix(thetas[i])
            # 将当前连杆的变换矩阵累乘到总变换矩阵上 (T_0^i = T_0^{i-1} * T_{i-1}^i)
            trans = np.dot(trans, link_trans) # np.dot 用于矩阵乘法
        return trans # 返回最终的 T_0^N 变换矩阵

    # 正向运动学计算
    def forward_kinematics(self, thetas):
        """
        计算给定关节角度下的末端执行器位姿（位置和 ZYZ 欧拉角表示的姿态）。
        :param thetas: 包含所有关节角度的列表或数组 (rad)
        :return: 一个列表 [x, y, z, alpha, beta, gamma]，表示末端执行器的位置和 ZYZ 欧拉角姿态。
        """
        # 计算总变换矩阵
        trans = self.transformation_matrix(thetas)
        # 从变换矩阵的最后一列提取末端执行器的位置 (x, y, z)
        x = trans[0, 3]
        y = trans[1, 3]
        z = trans[2, 3]

        # 使用 euler_angle 方法从变换矩阵计算 ZYZ 欧拉角
        alpha, beta, gamma = self.euler_angle(thetas) # 或者可以直接传入 trans
        # 返回包含位置和姿态的列表
        return [x, y, z, alpha, beta, gamma]

    # 从变换矩阵计算 ZYZ 欧拉角
    def euler_angle(self, thetas):
        """
        从给定的关节角度计算得到的末端执行器姿态的 ZYZ 欧拉角。
        注意：欧拉角的计算可能存在奇点和多解问题。这里的实现尝试处理一些情况，但可能不完全鲁棒。
        :param thetas: 包含所有关节角度的列表或数组 (rad)
        :return: alpha, beta, gamma 三个 ZYZ 欧拉角 (rad)
        """
        # 首先计算总变换矩阵
        trans = self.transformation_matrix(thetas)
        # R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]] = trans[0:3, 0:3]
        # ZYZ 欧拉角公式:
        # beta = atan2(sqrt(r13^2 + r23^2), r33)  (常用形式)
        # alpha = atan2(r23 / sin(beta), r13 / sin(beta)) => atan2(r23, r13)
        # gamma = atan2(r32 / sin(beta), -r31 / sin(beta)) => atan2(r32, -r31)

        # 代码中的实现方式略有不同，但目标是求解 ZYZ 欧拉角
        # 计算 alpha (绕 Z 轴的第一个旋转角)
        alpha = math.atan2(trans[1, 2], trans[0, 2]) # atan2(r23, r13)

        # 这部分逻辑试图将 alpha 限制在 [-pi/2, pi/2] 内，这对于标准的 ZYZ 定义可能不是必须的，
        # 并且可能在某些情况下引入错误。标准的 atan2 结果范围是 [-pi, pi]。
        # 保留原始逻辑，但需注意其可能的影响。
        if not (-np.pi / 2 <= alpha <= np.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) + np.pi
        if not (-np.pi / 2 <= alpha <= np.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) - np.pi

        # 计算 beta (绕新的 Y' 轴的旋转角)
        # beta = atan2(r13*cos(alpha) + r23*sin(alpha), r33)
        # 因为 cos(alpha) = r13 / sqrt(r13^2+r23^2) and sin(alpha) = r23 / sqrt(r13^2+r23^2)
        # 所以 r13*cos(alpha) + r23*sin(alpha) = (r13^2 + r23^2) / sqrt(r13^2+r23^2) = sqrt(r13^2+r23^2)
        # 这与常用公式 atan2(sqrt(r13^2 + r23^2), r33) 等价
        beta = math.atan2(
            trans[0][2] * np.cos(alpha) + trans[1][2] * np.sin(alpha),
            trans[2][2])

        # 计算 gamma (绕最终的 Z'' 轴的旋转角)
        # gamma = atan2(-r11*sin(alpha) + r21*cos(alpha), -r12*sin(alpha) + r22*cos(alpha))
        # 这对应于 atan2(r32, -r31)
        gamma = math.atan2(
            -trans[0][0] * np.sin(alpha) + trans[1][0] * np.cos(alpha),
            -trans[0][1] * np.sin(alpha) + trans[1][1] * np.cos(alpha))

        return alpha, beta, gamma # 返回计算得到的三个欧拉角

    # 逆向运动学计算
    def inverse_kinematics(self, ref_ee_pose):
        """
        使用基于雅可比伪逆的迭代方法计算逆运动学。
        给定目标末端执行器位姿，求解对应的关节角度。
        注意：这是一个数值解法，不保证找到解，可能陷入局部最优，且收敛性依赖于初始猜测和步长。
              其使用的雅可比与姿态误差的映射关系 (`K_alpha`) 可能需要根据具体应用场景仔细推导或调整。
        :param ref_ee_pose: 期望的末端执行器位姿列表 [x, y, z, alpha, beta, gamma]
        :return: 计算得到的关节角度列表 [theta1, theta2, ..., thetaN] (rad)
        """
        # 初始化关节角度猜测值（例如，全零）
        thetas = np.array([0.0] * len(self.link_list)) # 假设与连杆数量相同
        # 迭代求解
        for cnt in range(500): # 设置最大迭代次数
            # 1. 计算当前关节角度下的末端执行器位姿 (正向运动学)
            ee_pose = self.forward_kinematics(thetas)
            # 2. 计算当前位姿与目标位姿之间的误差
            diff_pose = np.array(ref_ee_pose) - np.array(ee_pose)
            # 可选：添加误差检查，如果误差足够小则提前退出循环
            # if np.linalg.norm(diff_pose) < tolerance:
            #     break

            # 3. 计算当前位姿下的几何雅可比矩阵 J_g
            basic_jacobian_mat = self.basic_jacobian(thetas)
            # 4. 获取当前姿态的欧拉角 (用于计算姿态误差对应的角速度)
            alpha, beta, gamma = self.euler_angle(thetas) # 或者从 ee_pose 中获取

            # 5. 计算将欧拉角速率映射到笛卡尔角速度的变换矩阵 K_zyz (或称 T_e, B)
            # omega = K_zyz * [alpha_dot, beta_dot, gamma_dot]
            K_zyz = np.array(
                [[0, -np.sin(alpha), np.cos(alpha) * np.sin(beta)],
                 [0,  np.cos(alpha), np.sin(alpha) * np.sin(beta)],
                 [1,  0,              np.cos(beta)]])

            # 6. 构建 6x6 变换矩阵 K_alpha，用于将位姿误差 [dx, dy, dz, d_alpha, d_beta, d_gamma]
            #    近似转换为期望的笛卡尔速度 [vx, vy, vz, wx, wy, wz]。
            #    这里的 K_alpha 用于将姿态误差（欧拉角差）转换为对应的期望角速度。
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz # 将 K_zyz 放入右下角 3x3 子矩阵

            # 7. 核心步骤：使用雅可比伪逆求解关节速度 (theta_dot)
            #    theta_dot = J_g^{+} * V_desired
            #    这里 V_desired 近似为 K_alpha * diff_pose (期望速度/位移)
            #    np.linalg.pinv 计算伪逆 J_g^{+}
            #    注意：这里的 K_alpha * diff_pose 是对期望笛卡尔速度的一种近似，并非严格推导。
            theta_dot = np.dot(
                np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha), # J_g^{+} * K_alpha
                np.array(diff_pose)) # * diff_pose

            # 8. 更新关节角度
            #    thetas_{k+1} = thetas_k + step_size * theta_dot
            thetas = thetas + theta_dot / 100. # 使用一个小的步长 (1/100) 控制更新幅度

        return thetas.tolist() # 返回最终计算得到的关节角度列表

    # 计算整个机械臂的几何雅可比矩阵
    def basic_jacobian(self, thetas):
        """
        计算整个机械臂在当前关节角度下的几何雅可比矩阵 J_g。
        J_g 关联关节速度和末端执行器的笛卡尔速度 (线速度和角速度)。
        V = [vx, vy, vz, wx, wy, wz]^T = J_g * [theta1_dot, theta2_dot, ..., thetaN_dot]^T
        :param thetas: 包含所有关节角度的列表或数组 (rad)
        :return: 6xN 的几何雅可比矩阵 (NumPy array)，N 是关节数量。
        """
        # 1. 计算当前末端执行器的位置 (只需要位置部分)
        ee_pos = self.forward_kinematics(thetas)[0:3]
        # 2. 初始化一个空列表，用于存储每个关节对应的雅可比列
        basic_jacobian_mat_cols = []
        # 3. 初始化累积变换矩阵为单位矩阵
        trans = np.identity(4)
        # 4. 遍历所有连杆（和关节）
        for i in range(len(self.link_list)):
            # 计算从基座到当前连杆 i 的坐标系的变换矩阵 T_0^i
            trans = np.dot(
                trans, self.link_list[i].transformation_matrix(thetas[i]))
            # 调用 Link 类的静态方法计算第 i 个关节的雅可比列 J_i
            # 需要传入 T_0^i 和末端执行器位置 ee_pos
            jacobian_col = self.link_list[i].basic_jacobian(trans, ee_pos)
            # 将计算得到的列添加到列表中
            basic_jacobian_mat_cols.append(jacobian_col)

        # 5. 将所有列组合成一个矩阵，并转置得到标准的 6xN 雅可比矩阵
        #    np.array(basic_jacobian_mat_cols) 会得到 Nx6 的矩阵，所以需要 .T 转置
        return np.array(basic_jacobian_mat_cols).T


# 主程序入口：当该脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 初始化 ROS 节点，命名为 "jacobian_test"
    rospy.init_node("jacobian_test")

    # 创建 ROS 发布器 (Publisher)
    # 发布末端执行器的笛卡尔位置 (使用 Point 消息类型，只包含 x, y, z)
    tool_pose_pub = rospy.Publisher("/tool_pose_cartesian", Point, queue_size=1)
    # 发布末端执行器的笛卡尔速度 (使用 Point 消息类型，只包含 vx, vy, vz)
    tool_velocity_pub = rospy.Publisher("/tool_velocity_cartesian", Point, queue_size=1)
    # 发布末端执行器的笛卡尔受力 (使用 Point 消息类型，只包含 fx, fy, fz)
    tool_force_pub = rospy.Publisher("/tool_force_cartesian", Point, queue_size=1)

    # 定义 Kinova Gen3 Lite 机械臂的 DH 参数 (示例)
    # 每一行代表一个连杆的 [alpha, a, d, theta_offset]
    # 单位：角度为弧度 (rad)，长度为米 (m)
    # 注意：这里的 theta_offset 可能包含了为了匹配 URDF 或实际机器人零位而进行的调整
    dh_params_list = np.array([[0,         0,          243.3/1000, 0],               # Link 1
                               [np.pi/2, 0,          10/1000,    np.pi/2],     # Link 2 (+pi/2 offset)
                               [np.pi,   280/1000,   0,          np.pi/2],     # Link 3 (+pi/2 offset)
                               [np.pi/2, 0,          245/1000,   np.pi/2],     # Link 4 (+pi/2 offset)
                               [np.pi/2, 0,          57/1000,    0],               # Link 5
                               [-np.pi/2,0,          235/1000,   -np.pi/2]])    # Link 6 (-pi/2 offset)

    # 使用定义的 DH 参数创建 NLinkArm 对象实例
    gen3_lite = NLinkArm(dh_params_list)

    # 进入 ROS 主循环，持续运行直到节点被关闭 (Ctrl+C)
    while not rospy.is_shutdown():
        # 等待并接收来自 "/my_gen3_lite/joint_states" 话题的最新 JointState 消息
        # 这会阻塞程序直到收到消息
        feedback = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState)

        # 从接收到的消息中提取前 6 个关节的数据
        thetas = feedback.position[0:6]    # 关节角度 (rad)
        velocities = feedback.velocity[0:6] # 关节速度 (rad/s)
        torques = feedback.effort[0:6]      # 关节力矩 (Nm)

        # --- 核心计算 ---
        # 1. 正向运动学：计算末端执行器位姿 [x, y, z, alpha, beta, gamma]
        tool_pose = gen3_lite.forward_kinematics(thetas)
        # 2. 计算几何雅可比矩阵 J_g
        J = gen3_lite.basic_jacobian(thetas)
        # 3. 计算末端执行器笛卡尔速度 V = J_g * theta_dot
        tool_velocity = J.dot(velocities) # [vx, vy, vz, wx, wy, wz]
        # 4. 计算末端执行器笛卡尔力/力矩 F = (J_g^T)^+ * tau (基于 tau = J_g^T * F)
        #    使用雅可比转置的伪逆来求解 F
        tool_force = np.linalg.pinv(J.T).dot(torques) # [fx, fy, fz, tx, ty, tz]

        # --- 准备并发布 ROS 消息 ---
        # 创建 Point 消息用于发布位置
        tool_pose_msg = Point()
        tool_pose_msg.x = tool_pose[0] # x 坐标
        tool_pose_msg.y = tool_pose[1] # y 坐标
        tool_pose_msg.z = tool_pose[2] # z 坐标

        # 创建 Point 消息用于发布线速度
        tool_velocity_msg = Point()
        tool_velocity_msg.x = tool_velocity[0] # vx
        tool_velocity_msg.y = tool_velocity[1] # vy
        tool_velocity_msg.z = tool_velocity[2] # vz

        # 创建 Point 消息用于发布线性力
        tool_force_msg = Point()
        tool_force_msg.x = tool_force[0] # fx
        tool_force_msg.y = tool_force[1] # fy
        tool_force_msg.z = tool_force[2] # fz

        # 发布消息到对应的 ROS 话题
        tool_pose_pub.publish(tool_pose_msg)
        tool_velocity_pub.publish(tool_velocity_msg)
        tool_force_pub.publish(tool_force_msg)

        # --- 在控制台打印信息 (可选) ---
        print(f"joint position: {np.round(thetas, 3)}") # 打印关节角度 (保留3位小数)
        print(f"joint velocity: {np.round(velocities, 3)}") # 打印关节速度
        print(f"joint torque: {np.round(torques, 3)}") # 打印关节力矩

        print(f"tool position (x,y,z): {np.round(tool_pose[0:3], 3)}") # 打印工具位置
        # print(f"tool orientation (a,b,g): {np.round(tool_pose[3:6], 3)}") # 可以取消注释以打印姿态
        print(f"tool velocity (vx,vy,vz): {np.round(tool_velocity[0:3], 3)}") # 打印工具线速度
        # print(f"tool angular velocity (wx,wy,wz): {np.round(tool_velocity[3:6], 3)}") # 可以取消注释以打印角速度
        print(f"tool force (fx,fy,fz): {np.round(tool_force[0:3], 3)}") # 打印工具受力
        # print(f"tool torque (tx,ty,tz): {np.round(tool_force[3:6], 3)}") # 可以取消注释以打印力矩
        print("-" * 20) # 打印分隔线

        # 控制循环频率
        # rate = rospy.Rate(10) # 10 Hz
        # rate.sleep()