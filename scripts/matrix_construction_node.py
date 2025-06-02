#!/usr/bin/env python3
import rospy
import math
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ee368_project.msg import ChessboardCorners
from ultralytics import YOLO

# 棋子类别映射
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
    -1: '0'
}

class ChessboardDetector:
    def __init__(self):
        rospy.init_node('chessboard_detector', anonymous=True)
        
        # 参数配置
        self.model_path = rospy.get_param('~model_path', '/home/slam/catkin_workspace/src/ee368_project/scripts/runs/chess_piece/chess_piece_exp1/weights/best.pt')
        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.corners_topic = rospy.get_param('~corners_topic', '/chessboard_corners')
        
        # 初始化组件
        self.bridge = CvBridge()
        self.grid_size = None
        self.latest_corners = None
        self.model = YOLO(self.model_path)
        
        # 创建发布者和订阅者
        self.board_pub = rospy.Publisher('/chess_board_matrix', String, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.corners_sub = rospy.Subscriber(self.corners_topic, ChessboardCorners, self.corners_callback)
        
        self.rate = rospy.Rate(10)  # 10 Hz
        self.corners_received = False  # 标志位

    def image_callback(self, msg):
        """处理图像消息"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.cv_image = cv_image  # 保存图像供检测使用
            rospy.loginfo("接收到图像，尺寸: %dx%d", cv_image.shape[1], cv_image.shape[0])
        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败: {str(e)}")

    def corners_callback(self, msg):
        """处理棋盘角点消息（仅处理第一次接收的数据）"""
        if self.corners_received:
            return  # 已接收过则直接返回
            
        try:
            # 关键修复：显式提取每个点的x/y坐标
            self.latest_corners = [
                (float(msg.top_left.x), float(msg.top_left.y)),  # 提取x,y分量
                (float(msg.top_right.x), float(msg.top_right.y)),
                (float(msg.bottom_left.x), float(msg.bottom_left.y)),
                (float(msg.bottom_right.x), float(msg.bottom_right.y))
            ]
            
            # 验证坐标类型
            for i, corner in enumerate(self.latest_corners):
                rospy.loginfo(f"角点{i+1}: x={corner[0]} ({type(corner[0])}), y={corner[1]} ({type(corner[1])})")
            
            self.grid_size = self.calculate_grid_size()
            self.corners_received = True  # 标记已接收
            
            # 取消订阅角点话题
            self.corners_sub.unregister()
            
        except Exception as e:
            rospy.logerr(f"处理角点失败: {str(e)}")

    def calculate_grid_size(self):
        """计算棋盘网格尺寸（基于首次接收的角点数据）"""
        if self.latest_corners is None:
            return None
            
        try:
            # 解包坐标（确保每个角点是(x,y)元组）
            (tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (al_x, al_y) = self.latest_corners
            
            # 确保数值类型
            tl_x = float(tl_x)
            tl_y = float(tl_y)
            tr_x = float(tr_x)
            tr_y = float(tr_y)
            bl_x = float(bl_x)
            bl_y = float(bl_y)
            
            # 计算实际尺寸（单位：米）
            width = math.hypot(tr_x - tl_x, tr_y - tl_y)
            height = math.hypot(bl_y - tl_y, bl_x - tl_x)
            
            # 标准棋盘规格（9列，10行）
            col_step = width / 8  # 9列需要8个间隔
            row_step = height / 9  # 10行需要9个间隔
            
            rospy.loginfo(f"棋盘宽度: {width:.2f}m, 高度: {height:.2f}m")
            rospy.loginfo(f"列间距: {col_step:.2f}m, 行间距: {row_step:.2f}m")
            return (col_step, row_step)
            
        except TypeError as e:
            rospy.logerr(f"类型错误: {str(e)}")
            return None
        except Exception as e:
            rospy.logerr(f"计算失败: {str(e)}")
            return None

    def split_into_regions(self):
        """生成9x10个区域（确保坐标正确）"""
        if self.latest_corners is None or self.grid_size is None:
            return []
        
        try:
            # 解包角点坐标（示例值，需根据实际数据调整）
            (tl_x, tl_y), (_, _), (bl_x, bl_y), _ = self.latest_corners
            col_step, row_step = self.grid_size
            
            # 强制转换为浮点数
            tl_x, tl_y, col_step, row_step = map(float, [tl_x, tl_y, col_step, row_step])
            
            regions = []
            for row in range(10):
                for col in range(9):
                    x1 = tl_x + col * col_step
                    y1 = tl_y + row * row_step
                    x2 = x1 + col_step
                    y2 = y1 + row_step
                    regions.append({
                        "top_left": (x1, y1),
                        "bottom_right": (x2, y2),
                        "center": (x1 + col_step/2, y1 + row_step/2)
                    })
            return regions
        except Exception as e:
            rospy.logerr(f"区域划分失败: {str(e)}")
            return []

    def detect_pieces(self, image):
        """执行棋子检测（确保输出格式正确）"""
        results = self.model(image, verbose=False)
        detections = []
        
        if not results:
            return detections
        
        for box in results[0].boxes:
            try:
                # 提取类别索引（兼容不同模型版本）
                cls = int(box.cls) if hasattr(box, 'cls') else int(box.class_id)
                
                # 转换坐标并计算中心点
                xyxy = box.xyxy.cpu().numpy().astype(int)
                xyxy = xyxy.squeeze()
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                # 映射类别到符号
                label = PIECE_CLASSES[cls]
                
                detections.append({
                    "label": label,
                    "center": (center_x, center_y),
                    "id": cls
                })
                rospy.loginfo(f"检测到棋子: {label} @ {center_x:.2f}, {center_y:.2f}")
                
            except Exception as e:
                rospy.logerr(f"检测失败: {str(e)}")
                
        return detections
    
    def map_piece_to_region(self, piece_center, regions):
        """将棋子中心点映射到区域索引（添加坐标校验）"""
        for idx, region in enumerate(regions):
            tl_x, tl_y = region["top_left"]
            br_x, br_y = region["bottom_right"]
            
            # 校验坐标是否在棋盘范围内（单位：米）
            if not (0 <= piece_center[0] <= 1.28 and 0 <= piece_center[1] <= 0.72):  # 假设棋盘尺寸1.28m×0.72m
                rospy.logwarn(f"棋子坐标越界: {piece_center}")
                return -1
                
            # 检查区域边界（含容差±5cm）
            if (tl_x - 0.05 <= piece_center[0] <= br_x + 0.05) and (tl_y - 0.05 <= piece_center[1] <= br_y + 0.05):
                return idx
        return -1

    def generate_region_matrix(self, detections, regions):
        """生成10x9的棋盘矩阵（添加调试信息）"""
        matrix = np.full((10, 9), -1, dtype=int)
        
        for det in detections:
            piece_center = det["center"]
            region_idx = self.map_piece_to_region(piece_center, regions)
            
            if region_idx != -1:
                # 打印映射结果
                rospy.logdebug(f"棋子 {det['label']} 映射到区域 {region_idx}")
                
                row = region_idx // 9
                col = region_idx % 9
                matrix[row][col] = self.get_symbol(det["label"])
            else:
                rospy.logwarn(f"棋子 {det['label']} 未映射到任何区域")
                
        # 打印完整矩阵
        rospy.loginfo(f"生成矩阵:{matrix}")
        return matrix

    def get_symbol(self, label):
        """将标签映射到棋子符号（根据示例调整）"""
        symbol_map = {
            "红帅": 'k', "黑将": 'K',
            "红仕": 'a', "黑士": 'A',
            "红相": 'b', "黑象": 'B',
            "红马": 'n', "黑马": 'N',
            "红车": 'r', "黑车": 'R',
            "红炮": 'c', "黑炮": 'C',
            "红兵": 'p', "黑卒": 'P',
            -1: f"{0}"  # 空位
        }
        return symbol_map.get(label, 0)

    def visualize_grid(self, image, regions):
        """在图像上绘制棋盘网格和检测到的棋子"""
        # 绘制网格线
        for i in range(1, 9):  # 8条垂直线
            x = regions[i*9]["top_left"][0]  # 每列的x坐标
            cv2.line(image, (int(x), int(regions[0]["top_left"][1])), 
                     (int(x), int(regions[-1]["bottom_right"][1])), (0, 255, 0), 1)
        
        for i in range(1, 10):  # 9条水平线
            y = regions[i]["top_left"][1]  # 每行的y坐标
            cv2.line(image, (int(regions[0]["top_left"][0]), int(y)), 
                     (int(regions[-1]["bottom_right"][0]), int(y)), (0, 255, 0), 1)
        
        # 绘制检测到的棋子
        for det in self.detections:
            x, y = int(det["center"][0]), int(det["center"][1])
            label = det["label"]
            symbol = self.get_symbol(label)
            
            # 绘制棋子位置
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            # 在棋子位置显示符号
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            cv2.putText(image, symbol, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return image

    def publish_region_matrix(self, matrix):
        """发布区域状态矩阵"""
        try:
            matrix_str = " ".join([" ".join(map(str, row)) for row in matrix])
            self.board_pub.publish(String(data=matrix_str))
            rospy.loginfo("Published region matrix")
        except Exception as e:
            rospy.logerr(f"发布区域矩阵失败: {str(e)}")

    def run(self):
        """主运行循环"""
        rospy.loginfo("Chessboard detector node started")
        while not rospy.is_shutdown():
            if self.latest_corners is None or self.grid_size is None:
                self.rate.sleep()
                continue
                
            try:
                # 获取当前图像
                cv_image = self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.image_topic, Image), "bgr8")
                
                # 生成区域划分（基于首次接收的角点数据）
                regions = self.split_into_regions()

                # 检测棋子
                detections = self.detect_pieces(cv_image)
                self.detections = detections  # 保存检测结果用于可视化

                # 生成区域矩阵
                region_matrix = self.generate_region_matrix(detections, regions)

                # 可视化网格和棋子
                visualized_image = self.visualize_grid(cv_image.copy(), regions)
                
                # 显示可视化图像（可选，用于调试）
                cv2.imshow("Chessboard Detection", visualized_image)
                cv2.waitKey(1)
                
                # 发布结果
                self.publish_region_matrix(region_matrix)

            except Exception as e:
                rospy.logerr(f"运行时错误: {str(e)}")
                
            self.rate.sleep()

if __name__ == '__main__':
    try:
        detector = ChessboardDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down chessboard detector node")