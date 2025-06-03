#!/usr/bin/env python3
import rospy
import math
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
from ee368_project.msg import ChessboardCorners

# 棋子类别映射字典（根据你的YOLO模型实际类别调整）
PIECE_CLASSES = {
    0: "红帅", 1: "红仕", 2: "红相", 3: "红马", 4: "红车",
    5: "红炮", 6: "红兵", 7: "黑将", 8: "黑士", 9: "黑象",
    10: "黑马", 11: "黑车", 12: "黑炮", 13: "黑卒"
}

CHESS_SYMBOLS = {
    "红帅": 'k', "黑将": 'K',
    "红仕": 'a', "黑士": 'A',
    "红相": 'b', "黑象": 'B',
    "红马": 'n', "黑马": 'N',
    "红车": 'r', "黑车": 'R',
    "红炮": 'c', "黑炮": 'C',
    "红兵": 'p', "黑卒": 'P',
    -1: '0'  # 未知类别占位符
}

class ChessboardDetector:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('matrix_construction', anonymous=True)
        
        # 创建发布者和订阅者
        self.bridge = CvBridge()
        self.board_pub = rospy.Publisher('/chess_board_matrix', String, queue_size=10)
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', 
            Image, 
            self.image_callback)
        self.corners_sub = rospy.Subscriber(
            '/chessboard_corners',
            ChessboardCorners,
            self.corners_callback
        )
        
        self.grid_size = None
        self.latest_corners = None
        self.latest_image = None  # 新增图像缓存
        self.rate = rospy.Rate(1)  # 1 Hz
        self.model = self.load_model("./runs/chess_piece/chess_piece_exp1/weights/best.pt")
        self.show_visualization = rospy.get_param("~show_visualization", True)
        self.cv_window_name = "Chessboard with Labels"
        
    def load_model(self, model_path):
        """加载并配置YOLO模型"""
        model = YOLO(model_path)
        model.fuse()  # 模型融合加速推理
        return model
    
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rospy.loginfo("接收到图像，尺寸: %dx%d", self.latest_image.shape[1], self.latest_image.shape[0])
            
        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败: {str(e)}")
            
    def corners_callback(self, msg):
        """处理棋盘角点消息"""
        try:
            # 提取四个角点坐标（单位：米）
            self.latest_corners = [
                (msg.top_left.x, msg.top_left.y),
                (msg.top_right.x, msg.top_right.y),
                (msg.bottom_left.x, msg.bottom_left.y),
                (msg.bottom_right.x, msg.bottom_right.y)
            ]
            rospy.loginfo("Received new corners: %s", self.latest_corners)
            
            # 首次收到角点时计算网格尺寸
            if self.grid_size is None:
                self.grid_size = self.calculate_grid_size()
                rospy.loginfo("Grid size calculated: %s", self.grid_size)
                
        except Exception as e:
            rospy.logerr("Error processing corners: %s", str(e))
            
    def calculate_grid_size(self):
        """计算棋盘网格尺寸"""
        if len(self.latest_corners) != 4:
            rospy.logerr("Invalid corners data")
            return None
            
        # 解析四个角点坐标
        (tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y) = self.latest_corners
        
        # 计算实际尺寸（单位：米）
        width = math.hypot(tr_x - tl_x, tr_y - tl_y)
        height = math.hypot(bl_y - tl_y, bl_x - tl_x)
        
        # 标准棋盘规格（9列，10行）
        num_cols = 9
        num_rows = 10
        
        # 计算每个格子的实际尺寸
        col_step = width / (num_cols - 1)
        row_step = height / (num_rows - 1)
        
        return (col_step, row_step)
    
    def detect_and_publish(self):
        """执行完整的棋盘检测和发布流程"""
        if self.latest_corners is None:
            rospy.logwarn("Waiting for corners...")
            return
        
        if self.grid_size is None:
            rospy.logwarn("Grid size not initialized")
            return
        
        # 执行棋子检测
        detections = self.detect_pieces()
        
        if not detections:
            rospy.logwarn("No pieces detected")
            return
        
        # 生成棋盘矩阵
        board_matrix = self.generate_board_matrix(detections)
        
        # 发布棋盘状态
        self.publish_board(board_matrix)
        
        # 可视化（如果有图像数据）
        if self.latest_image is not None:
            self.visualize_board_label(
                self.latest_image, 
                board_matrix, 
                detections, 
                self.grid_size
            )
        
    def detect_pieces(self, img):
        """检测图像中的棋子"""
        results = self.model(img, size=img.shape[:2][::-1])  # 保持长宽比推理
        
        detections = []
        for box in results[0].boxes:
            # 过滤低置信度检测
            if box.conf[0] < 0.5:
                continue
                
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            
            # 提取边界框和中心点
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x_center = (xyxy[0] + xyxy[2]) // 2
            y_center = (xyxy[1] + xyxy[3]) // 2
            
            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "bbox": xyxy.tolist(),
                "center": (x_center, y_center)
            })
        
        return detections
    
    def generate_board_matrix(self, detections):
        """生成棋盘状态矩阵"""
        if self.latest_corners is None or self.grid_size is None:
            return np.full((10,9), -1, dtype=int)
        
        # 创建空棋盘矩阵
        board = np.full((10,9), -1, dtype=int)
        
        for det in detections:
            x, y = det["center"]
            
            # 计算行列索引
            col = int(round((x - self.latest_corners[0][0]) / self.grid_size[0]))
            row = int(round((y - self.latest_corners[0][1]) / self.grid_size[1]))
            
            # 边界检查
            if 0 <= row < 10 and 0 <= col < 9:
                label_idx = self.get_symbol_index(det["label"])
                board[row][col] = label_idx
                
        return board
    
    def get_symbol_index(self, label):
        """获取棋子对应的符号索引"""
        try:
            return list(PIECE_CLASSES.keys())[list(PIECE_CLASSES.values()).index(label)]
        except ValueError:
            rospy.logwarn(f"Unknown piece label: {label}")
            return -1
    
    def publish_board(self, matrix):
        """发布棋盘状态"""
        try:
            ros_data = str(matrix).replace("'", '"')
            self.board_pub.publish(String(data=ros_data))
            rospy.loginfo("Published board matrix")
        except Exception as e:
            rospy.logerr("Failed to publish board: %s", str(e))
            
    def visualize_board_label(self, img, board_matrix, detections, grid_size):
        """可视化棋盘和棋子标签（仅修改此函数）"""
        height, width = img.shape[:2]
        
        # 绘制棋盘网格
        for i in range(1, 10):
            cv2.line(img, (0, i*grid_size), (width, i*grid_size), (0, 255, 0), 1)
        for j in range(1, 9):
            cv2.line(img, (j*grid_size, 0), (j*grid_size, height), (0, 255, 0), 1)
        
        # 绘制棋子标签（动态计算行列）
        for det in detections:
            x, y = det["center"]
            label = det["label"]
            
            # 动态计算行列（反转y轴方向）
            row = int(round(y / grid_size))  # ✅ 使用传入的 min_y
            col = int(round(x / grid_size))  # ✅ 根据网格尺寸计算列号
            
            # 边界检查
            row = max(0, min(row, 9))
            col = max(0, min(col, 8))
            
            # 定义文本位置（棋子中心上方）
            text_x = int(x)
            text_y = int(y - 15)
            
            # 设置颜色（红方黑色文字，黑方白色文字）
            color = (0, 0, 255) if "红" in label else (255, 255, 255)
            
            # 绘制文本背景
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, 
                         (text_x - text_width//2, text_y - text_height - 5),
                         (text_x + text_width//2, text_y + 5),
                         (0, 0, 0), -1)
            
            # 绘制文本
            cv2.putText(img, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 标注棋盘坐标
        for row in range(10):
            for col in range(9):
                if board_matrix[row, col] > 0:
                    x = col * grid_size + grid_size // 2
                    y = row * grid_size + grid_size // 2
                    cv2.putText(img, str(board_matrix[row, col]),
                               (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow(self.cv_window_name, img)
        cv2.waitKey(1)  # 保持窗口响应

    def run(self):
        """主运行循环"""
        rospy.loginfo("Chessboard detector node started")
        while not rospy.is_shutdown():
            self.detect_and_publish()
            self.rate.sleep()

if __name__ == '__main__':
    detector = None  # 初始化detector变量
    try:
        detector = ChessboardDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down chessboard detector node")
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")
    finally:
        # 确保关闭所有OpenCV窗口
        if detector is not None and hasattr(detector, 'show_visualization') and detector.show_visualization:
            cv2.destroyAllWindows()