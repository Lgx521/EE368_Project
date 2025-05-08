import pyrealsense2 as rs
import numpy as np
import cv2

# --- 配置 ---
pipeline = rs.pipeline()
config = rs.config()

# 获取设备产品线，以便为不同的设备（如D400系列或L500系列）设置良好的默认值
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
try:
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
except RuntimeError as e:
    print(f"无法解析 pipeline: {e}")
    print("请确保 RealSense 设备已连接并且驱动已正确安装。")
    exit()


# 根据需要配置流
# 对于某些设备，可能需要检查支持的分辨率和格式
# 例如: config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#       config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 尝试使用常见的 640x480 分辨率
WIDTH, HEIGHT = 640, 480
FPS = 30

try:
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
except RuntimeError as e:
    print(f"无法启用流 (尝试 640x480 @ 30FPS): {e}")
    print("设备可能不支持此配置。尝试其他分辨率/FPS。")
    # 你可以在这里尝试其他配置，或者列出支持的配置
    # 例如，查找支持的配置：
    # for sensor in device.query_sensors():
    #     for profile in sensor.get_stream_profiles():
    #         print(profile)
    exit()


# --- 启动 Pipeline ---
profile = pipeline.start(config)

# 获取深度传感器的深度缩放因子 (米/单位)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度缩放因子: {depth_scale} (1 unidade de profundidade = {depth_scale:.4f} metros)")

# 创建一个对齐对象
# rs.align 允许我们将深度帧与其他帧对齐
# "align_to" 是我们计划将深度帧对齐到的流类型
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 等待一对连贯的帧: 深度和颜色
        frames = pipeline.wait_for_frames()

        # 将深度帧与颜色帧对齐
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame() # 从对齐后的帧集中获取颜色帧

        if not depth_frame or not color_frame:
            print("丢失帧...")
            continue

        # --- 将图像转换为 NumPy 数组 ---
        # 颜色图像
        color_image = np.asanyarray(color_frame.get_data())

        # 深度图像
        depth_image = np.asanyarray(depth_frame.get_data()) # 这是原始的16位深度数据

        # --- 准备用于显示的深度图像 ---
        # 1. 应用颜色映射使深度变化更易于观察
        #    首先将16位深度图转换为8位图 (0-255 范围)
        #    你可以通过乘以一个alpha因子并裁剪，或者使用cv2.normalize
        #    这里我们使用 cv2.convertScaleAbs 进行简单缩放，alpha 值需要调整以获得最佳视觉效果
        #    alpha = 0.03 大约对应于将 8 米的最大距离映射到 255 (255 / (8 / depth_scale))
        #    更通用的方法是归一化:
        #    depth_image_8bit = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 使用 cv2.convertScaleAbs 缩放深度值到 0-255 (8位)
        # alpha=0.03 假设深度值在几千的范围内 (毫米).
        # 比如，如果最大可见距离是5米 (5000mm), alpha = 255/5000 = 0.051
        # 如果最大可见距离是8米 (8000mm), alpha = 255/8000 = 0.031
        # 你也可以选择一个固定的裁剪范围来归一化，例如0.5米到8米
        # depth_clipped = np.clip(depth_image, 500, 8000) # 裁剪0.5米到8米
        # depth_display_gray = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_colormap = cv2.applyColorMap(depth_display_gray, cv2.COLORMAP_JET)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # alpha=0.03 假设深度值以毫米为单位，最大可视化距离约为8米（255/0.03 ≈ 8500mm）
        # 如果深度单位不同或范围差异大，这个alpha值需要调整

        # --- 显示图像 ---
        # 将颜色图像和深度颜色映射图像水平堆叠（如果它们高度相同）
        # images = np.hstack((color_image, depth_colormap))
        # cv2.namedWindow('RealSense Streams', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense Streams', images)

        # 或者分开显示
        cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Color', color_image)

        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Depth', depth_colormap)

        key = cv2.waitKey(1)
        # 按 ESC 或 'q' 键关闭图像窗口
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # 停止流
    pipeline.stop()
    print("Pipeline 已停止.")
    cv2.destroyAllWindows() # 确保所有OpenCV窗口都已关闭