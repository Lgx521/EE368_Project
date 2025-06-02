#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import pyrealsense2 as rs
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header # 主要用于Image消息中的header

# 可配置参数
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FPS = 10
TOPIC_NAME = '/camera/color/image_raw' # 标准的ROS图像话题名称
FRAME_ID = 'camera_color_optical_frame' # 标准的ROS TF frame ID

def main():
    # 1. 初始化ROS节点
    rospy.init_node('camera_publisher', anonymous=True)
    rospy.loginfo("ROS RealSense Camera Publisher Node Started")

    # 2. 创建ROS图像发布者
    image_pub = rospy.Publisher(TOPIC_NAME, Image, queue_size=10)
    
    # 3. 初始化RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 检查是否有RealSense设备连接
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        rospy.logerr("No RealSense devices found. Exiting.")
        return

    # 配置彩色图像流
    # 注意：某些RealSense相机可能不支持所有分辨率/帧率组合
    # BGR8是ROS中常用的格式，cv_bridge也很好处理
    try:
        config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
    except RuntimeError as e:
        rospy.logerr(f"Failed to enable color stream: {e}")
        rospy.logerr("Please check if the camera supports {IMG_WIDTH}x{IMG_HEIGHT} @ {FPS}FPS with BGR8 format.")
        rospy.logerr("You might need to adjust IMG_WIDTH, IMG_HEIGHT, or FPS.")
        return

    rospy.loginfo(f"Configured RealSense: Color stream {IMG_WIDTH}x{IMG_HEIGHT} @ {FPS} FPS, BGR8 format")

    # 4. 启动Pipeline
    try:
        profile = pipeline.start(config)
        rospy.loginfo("RealSense pipeline started.")
    except RuntimeError as e:
        rospy.logerr(f"Failed to start RealSense pipeline: {e}")
        return

    # 获取彩色图像流的内参 (如果需要发布CameraInfo)
    # color_profile = profile.get_stream(rs.stream.color)
    # color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    # rospy.loginfo(f"Color Intrinsics: {color_intrinsics}")
    # TODO: 如果需要，可以基于这些内参构建并发布 sensor_msgs/CameraInfo 消息

    # 5. 设置ROS循环频率
    # 理论上可以和相机帧率一致，但为了避免ROS处理瓶颈，可以略低或由wait_for_frames控制
    rate = rospy.Rate(FPS) 

    try:
        while not rospy.is_shutdown():
            # a. 等待并获取帧
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000) # 设置超时以避免永久阻塞
            except RuntimeError as e:
                rospy.logwarn_throttle(5.0, f"Timeout or error waiting for frames: {e}. Retrying...")
                # 发生错误时可以尝试重新连接或简单跳过
                # pipeline.stop()
                # profile = pipeline.start(config)
                continue 
            
            color_frame = frames.get_color_frame()

            if not color_frame:
                rospy.logwarn_throttle(1.0, "No color frame received in this set.")
                continue

            # b. 将RealSense帧数据转换为NumPy数组
            # pyrealsense2返回的数据已经是NumPy friendly的，但用asanyarray确保
            # 对于bgr8格式，数据已经是BGR顺序
            color_image_np = np.asanyarray(color_frame.get_data())

            # c. 创建ROS Image消息
            ros_image_msg = Image()
            ros_image_msg.header.stamp = rospy.Time.now() # 使用ROS当前时间戳
            ros_image_msg.header.frame_id = FRAME_ID
            
            ros_image_msg.height = color_image_np.shape[0]
            ros_image_msg.width = color_image_np.shape[1]
            
            # 编码: bgr8 表示 8位BGR彩色图像
            # 如果RealSense配置为rs.format.rgb8, 这里就用 "rgb8"
            ros_image_msg.encoding = "bgr8" 
            
            ros_image_msg.is_bigendian = 0 # 现代CPU通常是小端
            
            # step: 每一行的字节数 = 宽度 * 通道数 * 每个通道的字节数
            # 对于bgr8: width * 3 channels * 1 byte/channel
            ros_image_msg.step = color_image_np.shape[1] * 3 
            
            # data: 图像的原始字节数据
            ros_image_msg.data = color_image_np.tobytes()

            # d. 发布图像消息
            image_pub.publish(ros_image_msg)

            # e. 按设定频率休眠 (如果wait_for_frames已经阻塞了足够时间，这里的sleep可能很短)
            # rate.sleep() # 使用wait_for_frames后，这个rate.sleep()可能不是主要控制因素了
                           # 但保留它可以防止CPU空转过快（如果wait_for_frames很快返回）

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node shutting down.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
    finally:
        # 6. 停止Pipeline并清理
        rospy.loginfo("Stopping RealSense pipeline...")
        pipeline.stop()
        rospy.loginfo("RealSense pipeline stopped. Node exited.")

if __name__ == '__main__':
    main()