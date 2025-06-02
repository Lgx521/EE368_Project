import numpy as np
from numpy import pi as pi
from scipy.spatial.transform import Rotation as R

# --- 配置参数 ---
# 你需要根据你的机械臂填写这些D-H参数
# 标准D-H参数: [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
# theta_i 将是我们的关节变量 q_i，theta_offset_i 是初始偏移
# 例如:
# dh_parameters = [
#     [np.pi/2, 0,       0.2,     0],        # Link 1 (alpha_0, a_0, d_1, theta_1_offset)
#     [0,       0.5,     0,       np.pi/2],  # Link 2 (alpha_1, a_1, d_2, theta_2_offset)
#     [np.pi/2, 0.1,     0,       0],        # Link 3 (alpha_2, a_2, d_3, theta_3_offset)
#     [-np.pi/2,0,       0.3,     0],        # Link 4 (alpha_3, a_3, d_4, theta_4_offset)
#     [np.pi/2, 0,       0,       0],        # Link 5 (alpha_4, a_4, d_5, theta_5_offset)
#     [0,       0,       0.1,     0]         # Link 6 (alpha_5, a_5, d_6, theta_6_offset)
# ]
# 请务必确认你使用的是标准D-H参数还是改进D-H参数，并相应调整dh_transform的实现

# ！！! 在这里填写你的D-H参数 ！！！
# 格式: list of lists, 每个子list是 [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
# alpha 和 theta_offset 应该用弧度 (np.pi)

dh_parameters = [
    [0,    0,      0.2433, 0    ], 
    [pi/2, 0,      0.03,   pi/2 ],  
    [pi,   0.28,   0.02,   pi/2 ],  
    [pi/2, 0,      0.245,  pi/2 ], 
    [pi/2, 0,      0.057,  pi   ],  
    [pi/2, 0,      0.235,  pi/2 ]  
]


# 可选：关节限制 (弧度)
# joint_limits = [
#     [-np.pi, np.pi],  # Joint 1 min, max
#     [-np.pi/2, np.pi/2], # Joint 2 min, max
#     # ... for all 6 joints
# ]
joint_limits = None # 如果不设置，则不检查

# --- 辅助函数 ---
def dh_transform_matrix(alpha, a, d, theta):
    """
    计算单个D-H变换矩阵 (标准D-H)
    T_i-1_i = Rot_x(alpha_{i-1}) * Trans_x(a_{i-1}) * Rot_z(theta_i) * Trans_z(d_i)
    但更常见的顺序是:
    T_i-1_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_{i-1}) * Rot_x(alpha_{i-1})
    这里我们使用后者，因为a_{i-1}和alpha_{i-1}通常与连杆i-1相关，theta_i和d_i与关节i相关
    """

    A = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])
    return A


def forward_kinematics(q, dh_params_table):
    """
    计算正运动学，返回末端执行器的变换矩阵 T_0_n 和所有中间变换矩阵
    q: 6个关节角度 (弧度)
    dh_params_table: DH参数表
    """
    T_cumulative = np.eye(4)
    transforms_list = [np.eye(4)] # Store T_0_0, T_0_1, ..., T_0_n

    if len(q) != len(dh_params_table):
        raise ValueError("Number of joint angles must match number of DH parameters rows.")

    for i in range(len(q)):
        alpha_im1, a_im1, d_i, theta_offset_i = dh_params_table[i]
        theta_i = q[i] + theta_offset_i # 实际关节角
        
        T_i_prev_i = dh_transform_matrix(alpha_im1, a_im1, d_i, theta_i)
        T_cumulative = T_cumulative @ T_i_prev_i
        transforms_list.append(np.copy(T_cumulative))
        
    return T_cumulative, transforms_list


def calculate_jacobian_numerical(q, dh_params_table, epsilon=1e-6):
    """
    通过数值差分计算雅可比矩阵 (6xN)，N是关节数
    J = [J_p; J_o]
    J_p: 位置雅可比 (3xN)
    J_o: 姿态雅可比 (3xN)
    """
    N = len(q)
    J = np.zeros((6, N))
    
    T_0_n_base, _ = forward_kinematics(q, dh_params_table)
    pos_base = T_0_n_base[:3, 3]
    rot_base = T_0_n_base[:3, :3]

    for i in range(N):
        q_plus = np.copy(q)
        q_plus[i] += epsilon
        T_0_n_plus, _ = forward_kinematics(q_plus, dh_params_table)
        pos_plus = T_0_n_plus[:3, 3]
        rot_plus = T_0_n_plus[:3, :3]

        q_minus = np.copy(q)
        q_minus[i] -= epsilon
        T_0_n_minus, _ = forward_kinematics(q_minus, dh_params_table)
        pos_minus = T_0_n_minus[:3, 3]
        rot_minus = T_0_n_minus[:3, :3]

        # 位置雅可比列
        J[:3, i] = (pos_plus - pos_minus) / (2 * epsilon)

        # 姿态雅可比列 (基于旋转向量的差分)
        # R_err_plus = rot_plus @ rot_base.T
        # R_err_minus = rot_minus @ rot_base.T
        # # 这里的rot_base可以换成 rot_plus 和 rot_minus 对应的姿态来得到更精确的差分
        # # 不过用一个中心点也可以

        # 使用 scipy Rotation 来获取旋转向量 (轴角表示中的轴乘以角度)
        # 旋转向量近似于小角度变化时的角速度向量 omega * dt
        # dR = R_plus @ R_minus.T
        # d_rot_vec = R.from_matrix(dR).as_rotvec()
        # J[3:, i] = d_rot_vec / (2 * epsilon)

        # 更精确的角速度计算方法: Skew(omega) = R_dot * R.T
        # omega = (rot_vec_plus - rot_vec_minus) / (2*epsilon) if rot_vecs are small
        # A more stable way for orientation Jacobian columns:
        # delta_R_plus = rot_plus @ rot_base.T
        # delta_R_minus = rot_minus @ rot_base.T
        # rot_vec_plus = R.from_matrix(delta_R_plus).as_rotvec()
        # rot_vec_minus = R.from_matrix(delta_R_minus).as_rotvec()
        # J[3:, i] = (rot_vec_plus - rot_vec_minus) / (2*epsilon)
        # This is not quite right. The orientation error is not just subtraction of rotvecs.
        
        # Using small angle approximation for dR = I + skew(d_omega)
        # R_plus @ R_base.T \approx I + skew(omega_plus_dt)
        # R_minus @ R_base.T \approx I + skew(omega_minus_dt)
        # omega_plus_dt = R.from_matrix(rot_plus @ rot_base.T).as_rotvec()
        # omega_minus_dt = R.from_matrix(rot_minus @ rot_base.T).as_rotvec()
        # J[3:, i] = (omega_plus_dt - omega_minus_dt) / (2*epsilon) # This is incorrect.

        # Correct numerical derivative for rotation:
        # (log(R_plus * R_base^-1) - log(R_minus * R_base^-1)) / (2*epsilon)
        # where log is the matrix logarithm mapping SO(3) to so(3) (skew-symmetric matrix)
        # then convert skew-symmetric to vector.
        # Or, more simply, perturb, get axis-angle of R_perturbed @ R_original.T
        # The axis-angle vector itself is the delta_orientation
        delta_rot_plus = rot_plus @ rot_base.T
        delta_rot_minus = rot_base @ rot_minus.T # Note the order for a symmetric derivative
        
        # Angle-axis vector of R1 * R2.T gives rotation from R2 to R1
        # So, rot_vec_plus is rotation from rot_base to rot_plus
        rot_vec_plus = R.from_matrix(delta_rot_plus).as_rotvec()
        # And rot_vec_minus is rotation from rot_minus to rot_base
        rot_vec_minus = R.from_matrix(delta_rot_minus).as_rotvec()
        
        # The column of the Jacobian is (d_omega_x, d_omega_y, d_omega_z)
        # J_omega_i = (rot_vec_plus + rot_vec_minus) / (2 * epsilon)
        # Let's use the definition where error_rot = R_target @ R_current.T
        # and the change in orientation w.r.t joint i is approximately axis_i * d_theta_i
        # So J_o[:, i] is the axis of rotation of joint i, projected onto base frame if using geometric.
        # For numerical:
        #   Consider that dx = J dq. If dq_i = epsilon, then dx = J[:,i] * epsilon
        #   So J[:,i] = dx / epsilon.
        #   For orientation, dx_ori = omega_error_vector
        #   omega_error = R.from_matrix(rot_plus @ rot_minus.T).as_rotvec() / (2 * epsilon)
        # This is also a common way:
        R_i_plus_perturbed = R.from_matrix(rot_plus)
        R_i_minus_perturbed = R.from_matrix(rot_minus)
        # Get the angular velocity vector corresponding to the change
        # (Log(R_plus * R_minus.inv) / (2*dt) )
        delta_R_for_jacobian = R_i_plus_perturbed * R_i_minus_perturbed.inv()
        J[3:, i] = delta_R_for_jacobian.as_rotvec() / (2 * epsilon)

    return J

def get_pose_error(T_target, T_current):
    """
    计算当前位姿与目标位姿之间的误差
    返回一个6x1的误差向量 [dx, dy, dz, d_roll, d_pitch, d_yaw_error_vec]
    这里的姿态误差是旋转向量 (轴角表示)
    """
    pos_target = T_target[:3, 3]
    rot_target = T_target[:3, :3]

    pos_current = T_current[:3, 3]
    rot_current = T_current[:3, :3]

    pos_error = pos_target - pos_current

    # 姿态误差: R_error = R_target * R_current^T
    # 这个旋转矩阵R_error表示从当前姿态到目标姿态所需的旋转
    # 它的轴角表示可以作为姿态误差向量
    R_error_mat = rot_target @ rot_current.T
    orientation_error_rotvec = R.from_matrix(R_error_mat).as_rotvec()
    
    # Ensure orientation_error_rotvec is a column vector for stacking
    return np.hstack((pos_error, orientation_error_rotvec)).reshape(6, 1)

def inverse_kinematics_numerical(
    target_pose_matrix, 
    initial_q, 
    dh_params_table,
    joint_limits_rad=None, # Optional: [[min1,max1], [min2,max2], ...]
    max_iterations=100, 
    pos_tolerance=1e-4,  # meters for position
    ori_tolerance=1e-3,  # radians for orientation error vector magnitude
    damping_factor_lambda=0.1, # For DLS
    step_size_alpha = 0.5 # Step size for updates
    ):
    """
    使用数值迭代法 (阻尼最小二乘 Damped Least Squares) 求解逆运动学
    target_pose_matrix: 目标末端执行器位姿 (4x4齐次变换矩阵)
    initial_q: 初始关节角度猜测 (弧度)
    dh_params_table: DH参数表
    joint_limits_rad: 关节限制 (弧度)
    """
    q_current = np.array(initial_q, dtype=float)
    num_joints = len(q_current)

    if len(dh_params_table) != num_joints:
        raise ValueError("DH parameters table rows must match number of joints in initial_q.")

    for iteration in range(max_iterations):
        T_current, _ = forward_kinematics(q_current, dh_params_table)
        
        error_vec = get_pose_error(target_pose_matrix, T_current)
        
        current_pos_error_norm = np.linalg.norm(error_vec[:3])
        current_ori_error_norm = np.linalg.norm(error_vec[3:])

        # print(f"Iter {iteration}: Pos Err: {current_pos_error_norm:.6f}, Ori Err: {current_ori_error_norm:.6f}")

        if current_pos_error_norm < pos_tolerance and current_ori_error_norm < ori_tolerance:
            print(f"Converged in {iteration+1} iterations.")
            return q_current, True

        J = calculate_jacobian_numerical(q_current, dh_params_table)
        
        # Damped Least Squares (DLS) or Levenberg-Marquardt
        # delta_q = J^T * (J * J^T + lambda^2 * I)^-1 * error_vec
        # For stability, use np.linalg.solve(A,b) for A*x=b instead of inv(A)*b
        lambda_sq = damping_factor_lambda**2
        # J_JT = J @ J.T # This is 6x6
        # term_to_invert = J_JT + lambda_sq * np.eye(J_JT.shape[0])
        # delta_q = J.T @ np.linalg.solve(term_to_invert, error_vec)
        
        # Simpler: Pseudoinverse with Tikhonov regularization (similar to DLS)
        # delta_q = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6)) @ error_vec
        # Or using pinv for Moore-Penrose pseudoinverse:
        # delta_q = np.linalg.pinv(J) @ error_vec # Standard pseudoinverse, can be unstable
        
        # Using SVD-based DLS for robustness
        # J = U S V^T
        # J_dls_inv = V @ np.diag(S / (S^2 + lambda_sq)) @ U^T
        try:
            # DLS update: delta_q = J_transpose * inv(J * J_transpose + lambda^2 * I) * error
            # This is a common formulation for redundant or singular cases. J is 6xN.
            # J_transpose is N x 6. J @ J_transpose is 6x6.
            # The update for q should be N x 1.
            # delta_q = J.T @ np.linalg.solve(J @ J.T + lambda_sq * np.eye(6), error_vec)

            # Alternative DLS formulation using J_hash = J^T (J J^T + lambda^2 I)^-1
            # More direct if J is fat (more joints than DoF needed)
            # If J is tall (N < 6), this might be an issue. Assume N=6 here.
            # J_hash = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(J.shape[0]))
            # delta_q = J_hash @ error_vec

            # Standard formulation for damped least squares using pseudoinverse idea:
            # For J dq = dx, dq = J_pinv dx
            # J_pinv_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(J.shape[0])) for J being m x n, m < n
            # J_pinv_dls = np.linalg.inv(J.T @ J + lambda_sq * np.eye(J.shape[1])) @ J.T for J being m x n, m > n
            # For m=n=6:
            delta_q = np.linalg.solve(J.T @ J + lambda_sq * np.eye(num_joints), J.T @ error_vec)

        except np.linalg.LinAlgError:
            print(f"Singularity or numerical issue encountered at iteration {iteration}.")
            # Fallback to a more heavily damped step or stop
            # Using a simpler pseudo-inverse with small damping if solve fails
            try:
                delta_q = np.linalg.pinv(J, rcond=lambda_sq) @ error_vec
            except np.linalg.LinAlgError:
                print("Failed even with pinv. Stopping.")
                return q_current, False


        q_current = q_current + step_size_alpha * delta_q.flatten() # flatten to make it 1D

        for i in range(num_joints):
             q_current[i] = np.arctan2(np.sin(q_current[i]), np.cos(q_current[i]))

        # Apply joint limits if provided
        if joint_limits_rad:
            for i in range(num_joints):
                q_current[i] = np.clip(q_current[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
                
        # Normalize angles (optional, but can help keep them in [-pi, pi] or [0, 2pi])
        # for i in range(num_joints):
        # q_current[i] = np.arctan2(np.sin(q_current[i]), np.cos(q_current[i]))


    print(f"IK solver did not converge within {max_iterations} iterations.")
    print(f"Final Pos Err: {current_pos_error_norm:.6f}, Ori Err: {current_ori_error_norm:.6f}")
    return q_current, False

# --- Main execution example ---
if __name__ == "__main__":
    print("6-DoF Inverse Kinematics Numerical Solver\n")
    
    num_dof = len(dh_parameters)
    if num_dof != 6:
        print(f"Warning: DH parameters define a {num_dof}-DOF arm, but script assumes 6-DOF for target pose.")

    # 1. 定义目标位姿 (4x4 齐次变换矩阵)
    #    位置 (x, y, z) 和姿态 (欧拉角 Roll, Pitch, Yaw -> 转换为旋转矩阵)
    target_position = np.array([0.3, 0.0, 0.0]) # meters
    target_orientation_euler = np.array([0, 0, 0]) # radians (roll, pitch, yaw)
    
    # 将欧拉角转换为旋转矩阵
    # scipy uses intrinsic 'xyz' for roll, pitch, yaw usually
    # or extrinsic 'XYZ'. Be careful with conventions.
    # Assuming 'xyz' intrinsic: Yaw around Z, then Pitch around new Y, then Roll around new X
    # Or 'ZYX' extrinsic: Roll around fixed X, then Pitch around fixed Y, then Yaw around fixed Z
    # Let's use a common 'xyz' sequence (Roll about x, Pitch about y, Yaw about z of the *body* frame)
    # which is equivalent to ZYX extrinsic.
    target_rotation_matrix = R.from_euler('xyz', target_orientation_euler, degrees=False).as_matrix()

    T_target = np.eye(4)
    T_target[:3, :3] = target_rotation_matrix
    T_target[:3, 3] = target_position

    print("Target Pose (Homogeneous Matrix):")
    print(T_target)
    print("\nTarget Position:", target_position)
    print("Target Orientation (Euler XYZ rad):", target_orientation_euler)
    print("-" * 30)

    # 2. 初始关节角度猜测 (弧度)
    
    # top_view_pos = np.array([30.66, 346.57, 72.23, 270.08, 265.45, 345.69]) * np.pi / 180

    top_view_pos = np.zeros(6)


    initial_joint_angles = np.array(top_view_pos) # example

    print("Initial Joint Angles (rad):", initial_joint_angles)
    print("-" * 30)

    # 3. (可选) 定义关节限制
    # example_joint_limits = [
    #     [-np.pi, np.pi], [-np.pi/2, np.pi/2], [-np.pi, np.pi],
    #     [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]
    # ]  # Radians
    example_joint_limits = None # No limits for this example

    # 4. 调用IK求解器
    print("Running Inverse Kinematics Solver...")
    solution_q, converged = inverse_kinematics_numerical(
        T_target,
        initial_joint_angles,
        dh_parameters,
        joint_limits_rad=example_joint_limits,
        max_iterations=200,
        pos_tolerance=1e-5, # tighter tolerance
        ori_tolerance=1e-4, # tighter tolerance
        damping_factor_lambda=0.05, # Can tune this
        step_size_alpha = 1.2 # Can tune this
    )

    print("-" * 30)
    if converged:
        print("IK Solution Found (Joint Angles in radians):")
        print(np.round(solution_q, 4)) # Print rounded for readability
        
        # 5. 验证解：将解代入正运动学，看是否接近目标位姿
        T_solution, _ = forward_kinematics(solution_q, dh_parameters)
        print("\nFK of Solution (Homogeneous Matrix):")
        print(np.round(T_solution, 4))

        final_pos_error = np.linalg.norm(T_target[:3,3] - T_solution[:3,3])
        
        R_err_mat_final = T_target[:3,:3] @ T_solution[:3,:3].T
        final_ori_error_angle = np.arccos(np.clip((np.trace(R_err_mat_final) - 1) / 2.0, -1.0, 1.0)) # Angle of rotation error

        print(f"\nFinal Position Error Norm: {final_pos_error:.6e} meters")
        print(f"Final Orientation Error Angle: {final_ori_error_angle:.6e} radians")

        if final_pos_error < 1e-4 and final_ori_error_angle < 1e-3:
             print("Verification: Solution is close to the target.")
        else:
             print("Verification: Solution has noticeable error from target. Check DH, tolerances, or target reachability.")

    else:
        print("IK Solution NOT Found or did not converge to desired tolerance.")
        print("Last attempted q:", np.round(solution_q, 4))
        T_final_attempt, _ = forward_kinematics(solution_q, dh_parameters)
        print("\nFK of last attempt:")
        print(np.round(T_final_attempt, 4))

    print("\n--- Script End ---")