import numpy as np
from numpy import pi as pi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# --- D-H 参数 ---
# 标准D-H参数: [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
dh_parameters = [
    [0,    0,      0.2433, 0    ],
    [pi/2, 0,      0.03,   pi/2 ],
    [pi,    0.28,  0.02,   pi/2 ], 
    [pi/2, 0,      0.245,  pi/2 ],
    [pi/2, 0,      0.057,  pi   ],
    [pi/2, 0,      0.235,  pi/2 ]
]

num_joints = len(dh_parameters)

joint_limits_rad = None # 或者定义如下

def dh_transform_matrix(alpha, a, d, theta):
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha), np.cos(alpha),   np.cos(alpha) * d],
        [0,              0,                           0,                          1]
    ])
    return T

def forward_kinematics(q, dh_params_table):
    T_cumulative = np.eye(4)
    if len(q) != len(dh_params_table):
        raise ValueError("Number of joint angles must match number of DH parameters rows.")
    for i in range(len(q)):
        alpha_im1, a_im1, d_i, theta_offset_i = dh_params_table[i]
        theta_i = q[i] + theta_offset_i
        T_i_prev_i = dh_transform_matrix(alpha_im1, a_im1, d_i, theta_i)
        T_cumulative = T_cumulative @ T_i_prev_i
    # 只返回末端执行器位姿，如果需要中间变换，可以修改
    return T_cumulative, [T_cumulative] # 保持与之前接口相似，但只返回最后一个

def get_pose_error(T_target, T_current):
    pos_target = T_target[:3, 3]
    rot_target = T_target[:3, :3]
    pos_current = T_current[:3, 3]
    rot_current = T_current[:3, :3]
    pos_error = pos_target - pos_current
    R_error_mat = rot_target @ rot_current.T
    orientation_error_rotvec = R.from_matrix(R_error_mat).as_rotvec()
    return np.hstack((pos_error, orientation_error_rotvec)).reshape(6, 1)

# --- 基于 SciPy Optimize 的逆运动学求解器 ---
def objective_function(q, target_pose_matrix, dh_params, pos_weight, ori_weight):
    """
    目标函数：最小化末端位姿误差的加权范数平方。
    q: 当前关节角度 (优化变量)
    target_pose_matrix: 目标4x4齐次变换矩阵
    dh_params: DH参数表
    pos_weight: 位置误差权重
    ori_weight: 姿态误差权重
    """
    T_current, _ = forward_kinematics(q, dh_params)
    error_vec = get_pose_error(target_pose_matrix, T_current) # (6,1)
    
    pos_error_sq_norm = np.linalg.norm(error_vec[:3])**2
    ori_error_sq_norm = np.linalg.norm(error_vec[3:])**2

    
    return pos_weight * pos_error_sq_norm + ori_weight * ori_error_sq_norm

def inverse_kinematics_optimizer(
    target_pose_matrix,
    initial_q,
    dh_params_table,
    joint_limits=None, # [[min1,max1], [min2,max2], ...]
    pos_weight=5.0,
    ori_weight=1.0,
    pos_tolerance=1e-5, # 用于检查最终解的容差
    ori_tolerance=1e-4, # 用于检查最终解的容差 (旋转向量范数)
    max_iterations=200,
    optimizer_ftol=1e-7 # 优化器的收敛容差 (目标函数值的变化)
    ):
    """
    使用 scipy.optimize.minimize (SLSQP) 求解逆运动学
    """
    num_dof = len(initial_q)
    
    # 准备关节限制给优化器 (bounds)
    # SLSQP也可以用constraints，但bounds更直接
    bounds_optimizer = None
    if joint_limits:
        bounds_optimizer = []
        for i in range(num_dof):
            # 确保下限小于上限
            min_val = joint_limits[i][0]
            max_val = joint_limits[i][1]
            if min_val > max_val:
                print(f"Warning: Joint {i} limits are invalid ({min_val} > {max_val}). Swapping them.")
                min_val, max_val = max_val, min_val
            bounds_optimizer.append((min_val, max_val))

    # 对初始关节角进行归一化，使其在[-pi, pi]范围内
    q_normalized_initial = np.array([np.arctan2(np.sin(val), np.cos(val)) for val in initial_q])

    print("Running Inverse Kinematics Solver (SciPy Optimize - SLSQP)...")
    result = minimize(
        objective_function,
        q_normalized_initial,
        args=(target_pose_matrix, dh_params_table, pos_weight, ori_weight),
        method='SLSQP',
        bounds=bounds_optimizer, # SLSQP 支持 bounds
        options={'disp': False, 'maxiter': max_iterations, 'ftol': optimizer_ftol} # disp:True可以看过程
    )

    solution_q = result.x
    # 再次归一化最终解的角度
    solution_q_normalized = np.array([np.arctan2(np.sin(val), np.cos(val)) for val in solution_q])

    # 检查收敛性和误差
    T_solution, _ = forward_kinematics(solution_q_normalized, dh_params_table)
    final_error_vec = get_pose_error(target_pose_matrix, T_solution)
    final_pos_error_norm = np.linalg.norm(final_error_vec[:3])
    final_ori_error_norm = np.linalg.norm(final_error_vec[3:])

    converged = result.success and (final_pos_error_norm < pos_tolerance) and (final_ori_error_norm < ori_tolerance)
    
    print(f"Optimizer status: {result.message}")
    print(f"Iterations: {result.nit}, Func evaluations: {result.nfev}")
    print(f"Final objective value: {result.fun:.4e}")

    if converged:
        print(f"Converged to solution with Pos Err: {final_pos_error_norm:.4e}, Ori Err: {final_ori_error_norm:.4e}")
    else:
        if not result.success:
            print("Optimizer reported failure to converge.")
        print(f"Solution may not meet tolerance. Pos Err: {final_pos_error_norm:.4e}, Ori Err: {final_ori_error_norm:.4e}")

    return solution_q_normalized, converged, final_pos_error_norm, final_ori_error_norm


# --- Main execution example ---
if __name__ == "__main__":
    print("6-DoF Inverse Kinematics SciPy Optimizer Solver\n")

    # 1. 定义目标位姿
    target_position = np.array([0.3, 0.1, 0.25]) # meters
    target_orientation_euler = np.array([0, pi, pi/4]) # radians (roll, pitch, yaw)
    
    target_rotation_matrix = R.from_euler('xyz', target_orientation_euler, degrees=False).as_matrix()
    T_target = np.eye(4)
    T_target[:3, :3] = target_rotation_matrix
    T_target[:3, 3] = target_position

    print("Target Pose (Homogeneous Matrix):")
    print(np.round(T_target, 4))
    print("-" * 30)

    # 2. 初始关节角度猜测 (弧度)
    # 保持与你之前类似的初始值
    top_view_pos_deg = np.array([0,0,0,0,0,0])
    initial_joint_angles_rad = top_view_pos_deg * np.pi / 180.0
    print("Initial Joint Angles (rad, unnormalized):", np.round(initial_joint_angles_rad,4))
    # 优化器内部会处理归一化或使用归一化后的初始值
    print("-" * 30)

    example_joint_limits = None # 测试时不使用严格限制

    # 3. 调用IK求解器
    solution_q, converged, pos_err, ori_err = inverse_kinematics_optimizer(
        T_target,
        initial_joint_angles_rad,
        dh_parameters,
        joint_limits=example_joint_limits,
        pos_weight=5.0,       # 位置误差的权重
        ori_weight=1.0,       # 姿态误差的权重 (可以调整，例如如果姿态更重要，给更大权重)
        pos_tolerance=1e-4,
        ori_tolerance=1e-4,
        max_iterations=300,  # 优化器的最大迭代次数
        optimizer_ftol=1e-8  # 优化器目标函数值的收敛阈值
    )

    print("-" * 30)
    if converged:
        print("IK Solution Found (Joint Angles in deg):")
        print(np.round(solution_q*180/pi, 2))
        
        T_solution, _ = forward_kinematics(solution_q, dh_parameters)
        print("\nFK of Solution (Homogeneous Matrix):")
        print(np.round(T_solution, 4))

        print(f"\nFinal Position Error Norm: {pos_err:.6e} meters")
        print(f"Final Orientation Error (rotvec norm): {ori_err:.6e} radians")
        print("Verification: Solution is close to the target.")
    else:
        print("IK Solution NOT Found or did not converge to desired tolerance.")
        print("Last attempted q (rad):", np.round(solution_q, 4))
        T_final_attempt, _ = forward_kinematics(solution_q, dh_parameters)
        print("\nFK of last attempt:")
        print(np.round(T_final_attempt, 4))
        print(f"\nError of last attempt: Pos Norm: {pos_err:.6e}, Ori Norm: {ori_err:.6e}")


    print("\n--- Script End ---")

