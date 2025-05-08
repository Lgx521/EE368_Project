import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

# --- 1. ArUco 初始化 ---
# 选择与你打印或生成的标记相同的字典
dictionary_name = aruco.DICT_6X6_250 # 确保与生成时一致
dictionary = aruco.getPredefinedDictionary(dictionary_name)

# 创建探测器参数
parameters = aruco.DetectorParameters()
# 你可以根据需要调整参数，例如:
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# 标记的实际物理尺寸（米）
MARKER_REAL_SIZE_METERS = 0.05 

# --- 2. RealSense 初始化 ---
pipeline = rs.pipeline()
config = rs.config()

# 获取设备产品线，以便我们可以配置特定于深度和颜色传感器的内容
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line)) # D400系列

# 配置颜色流
# 你可以根据需要调整分辨率和帧率
# 常见的 D435 彩色分辨率: 640x480, 1280x720, 1920x1080
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30
config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)

# 启动流
profile = pipeline.start(config)

# 获取颜色流的内参
color_profile = profile.get_stream(rs.stream.color)
intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()

# 将 RealSense 内参转换为 OpenCV 格式
camera_matrix = np.array([
    [intrinsics_color.fx, 0, intrinsics_color.ppx],
    [0, intrinsics_color.fy, intrinsics_color.ppy],
    [0, 0, 1]
], dtype=np.float32)

# RealSense D400 系列通常畸变系数为0或使用特定模型。
# estimatePoseSingleMarkers 期望的是 Plumb Bob 模型 (k1, k2, p1, p2, k3)
# intrinsics_color.coeffs 通常对于D400系列是 [0,0,0,0,0]
dist_coeffs = np.array(intrinsics_color.coeffs, dtype=np.float32)
if dist_coeffs is None or len(dist_coeffs) == 0:
    dist_coeffs = np.zeros((5,1), dtype=np.float32) # 假设无畸变或使用默认的0值
    print("Warning: Distortion coefficients not found or empty, assuming zero distortion.")
elif len(dist_coeffs) != 5 and len(dist_coeffs) != 4 and len(dist_coeffs) != 8 and len(dist_coeffs) != 12 and len(dist_coeffs) != 14:
    # OpenCV's estimatePoseSingleMarkers expects 4, 5, 8, 12 or 14 elements
    print(f"Warning: Unexpected number of distortion coefficients ({len(dist_coeffs)}). Using zero distortion.")
    dist_coeffs = np.zeros((5,1), dtype=np.float32)


print("RealSense Camera Intrinsics (fx, fy, cx, cy):")
print(f"  fx: {intrinsics_color.fx:.2f}, fy: {intrinsics_color.fy:.2f}")
print(f"  cx: {intrinsics_color.ppx:.2f}, cy: {intrinsics_color.ppy:.2f}")
print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
print("Press 'q' to quit...")

try:
    while True:
        # --- 3. 从 RealSense 获取帧 ---
        frames = pipeline.wait_for_frames()
        color_frame_rs = frames.get_color_frame()

        if not color_frame_rs:
            print("No color frame received")
            continue

        # 将 RealSense 帧转换为 NumPy 数组 (OpenCV 格式)
        # RealSense 输出的是 BGR 格式，与 OpenCV 兼容
        color_image = np.asanyarray(color_frame_rs.get_data())

        # 将帧转换为灰度图像
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # --- 4. 检测 ArUco 标记 ---
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame,
            dictionary,
            parameters=parameters
        )

        # --- 5. 处理和绘制结果 ---
        display_image = color_image.copy() # 操作副本以保留原始彩色图像

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)

            # 估计每个标记的姿态
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_REAL_SIZE_METERS,
                camera_matrix,
                dist_coeffs
            )

            # 绘制每个标记的坐标轴
            for i in range(len(ids)):
                try:
                    # cv2.drawFrameAxes 是较新的函数 (OpenCV 4.7.0+)
                    # 老版本用 aruco.drawAxis(display_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_REAL_SIZE_METERS / 2)
                    cv2.drawFrameAxes(display_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_REAL_SIZE_METERS / 2)
                except cv2.error as e:
                    print(f"Error drawing axes for marker ID {ids[i]}: {e}")
                    # 尝试使用旧版 aruco.drawAxis 作为后备
                    try:
                        aruco.drawAxis(display_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_REAL_SIZE_METERS / 2)
                    except Exception as e_axis:
                        print(f"Fallback aruco.drawAxis also failed: {e_axis}")


                # 打印姿态信息 (可选)
                # rvec_formatted = ", ".join([f"{x:.3f}" for x in rvecs[i][0]])
                # tvec_formatted = ", ".join([f"{x:.3f}" for x in tvecs[i][0]])
                # print(f"Marker ID: {ids[i][0]:<3} | Rvec: [{rvec_formatted}] | Tvec: [{tvec_formatted}] (m)")

        else:
            cv2.putText(display_image, "No ArUco markers detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示结果帧
        cv2.imshow("RealSense ArUco Detection", display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'): # 按 's' 保存相机内参和畸变系数
            np.savez("realsense_intrinsics.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            print("Camera intrinsics saved to realsense_intrinsics.npz")


finally:
    # --- 6. 清理 ---
    print("Stopping RealSense pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")