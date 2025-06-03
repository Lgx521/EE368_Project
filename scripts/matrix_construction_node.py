#!/usr/bin/env python3
import rospy
import math
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ee368_project.msg import ChessboardPixelCorners
from ee368_project.msg import RegionMatrix 
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
    -1: f'{0}'
}

class ChessboardDetector:
    def __init__(self):
        rospy.init_node('chessboard_detector', anonymous=True)
        
        # 参数配置
        self.model_path = rospy.get_param('~model_path', './src/ee368_project/scripts/runs/chess_piece/chess_piece_exp1/weights/best.pt')
        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.corners_topic = rospy.get_param('~corners_topic', '/chessboard_pixel_corners')
        
        # 初始化组件
        self.bridge = CvBridge()
        self.grid_size = None
        self.latest_corners = None
        self.model = YOLO(self.model_path)
        
        # 创建发布者和订阅者
        self.board_pub = rospy.Publisher('/chess_board_matrix', RegionMatrix, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.corners_sub = rospy.Subscriber(self.corners_topic, ChessboardPixelCorners, self.corners_callback)
        
        self.rate = rospy.Rate(10)  # 10 Hz
        self.corners_received = False  # 标志位

    def image_callback(self, msg):
        """处理图像消息"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.cv_image = cv_image  # 保存图像供检测使用
        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败: {str(e)}")

    def corners_callback(self, msg):
        """处理棋盘角点消息（仅处理第一次接收的数据）"""
        if self.corners_received:
            return  # 已接收过则直接返回
            
        try:
            # 关键修复：显式提取每个点的x/y坐标
            self.latest_corners = [
                (float(msg.top_left_px.x), float(msg.top_left_px.y)),  # 提取x,y分量
                (float(msg.top_right_px.x), float(msg.top_right_px.y)),
                (float(msg.bottom_left_px.x), float(msg.bottom_left_px.y)),
                (float(msg.bottom_right_px.x), float(msg.bottom_right_px.y))
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
            (tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y) = self.latest_corners
            
            # 确保数值类型
            tl_x = float(tl_x)
            tl_y = float(tl_y)
            tr_x = float(tr_x)
            tr_y = float(tr_y)
            bl_x = float(bl_x)
            bl_y = float(bl_y)
            br_x = float(br_x)
            br_y = float(br_y)

            
            # 计算实际尺寸（单位：米）
            width = (tr_y - tl_y + br_y - bl_y) / 2
            height = (tl_x - bl_x + tr_x - br_x)/ 2
            
            # 标准棋盘规格（9列，10行）
            col_step = width / 8  # 9列需要8个间隔
            row_step = height / 9  # 10行需要9个间隔
            
            rospy.loginfo(f"棋盘宽度: {width:.2f}pixel, 高度: {height:.2f}pixel")
            rospy.loginfo(f"列间距: {col_step:.2f}pixel, 行间距: {row_step:.2f}pixel")
            
            regions = []
            for row in range(10):  # 10行
                for col in range(9):  # 9列
                    # 关键修正：向负轴方向偏移半个间隔（确保中心对齐）
                    y_offset = -col_step / 2  # X轴向左偏移半个间隔
                    x_offset = row_step / 2  # Y轴向上偏移半个间隔
                    
                    y1 = bl_y + col * col_step + y_offset
                    x1 = bl_x + row * row_step + x_offset
                    y2 = y1 + col_step
                    x2 = x1 - row_step
                    
                    region = {
                        "top_left": (x1, y1),
                        "bottom_right": (x2, y2),
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2)  # 交叉点作为中心
                    }
                    regions.append(region)
            return regions
            
        except TypeError as e:
            rospy.logerr(f"类型错误: {str(e)}")
            return None
        except Exception as e:
            rospy.logerr(f"计算失败: {str(e)}")
            return None
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
        """将棋子中心点映射到区域索引（左下角为原点）"""
        idx = -1
        for i, region in enumerate(regions):
            tl_x, tl_y = region["top_left"]
            br_x, br_y = region["bottom_right"]
                
            # 检查区域边界（含容差±30pixels）
            if (br_x - 10 <= piece_center[0] <= tl_x + 10 and 
                tl_y - 10 <= piece_center[1] <= br_y + 10):
                idx = i
        return idx

    def generate_region_matrix(self, detections, regions):
        """生成10x9的棋盘矩阵（存储字符串）"""
        # 修改 dtype 为 object 或 str，以便存储字符串
        matrix = np.full((10, 9), -1, dtype=object) 
        
        for det in detections:
            piece_center = det["center"]
            region_idx = self.map_piece_to_region(piece_center, regions)
            
            if region_idx != -1:
                # 打印映射结果
                rospy.logdebug(f"棋子 {det['label']} 映射到区域 {region_idx}")
                
                # 关键修正：正确的行和列计算
                row = 9 - (region_idx // 9)  # 行数是10，所以用//9
                col = region_idx % 9   # 列数是9，所以用%9
                
                # 确保行和列在有效范围内
                if 0 <= row < 10 and 0 <= col < 9:
                    # 确保 self.get_symbol 返回的是字符串（如果返回数字，可以 str() 转换）
                    symbol = self.get_symbol(det["label"])
                    matrix[row][col] = symbol  # 现在可以存储字符串
            else:
                # 没有棋子：应该放置"0"（空位置）
                # 但这里需要确认是否有逻辑覆盖
                pass
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
            -1: f'{0}'  # 空位
        }
        return symbol_map.get(label, 0)

    def replace_spaces_with_zero(self, matrix):
        """将矩阵中的空格 ' ' 替换为字符串 '0'"""
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == -1:  # 检查是否是空格
                    matrix[i][j] = f'{0}'    # 直接替换为字符串 '0'
        return matrix
    
    def visualize_grid(self, image, regions):
        # 复制图像以避免修改原始图像
        visualized_image = image.copy()
        
        # 获取图像尺寸
        height, width = visualized_image.shape[:2]
        
        # 绘制水平线（行分隔线）
        for row in range(1, 10):  # 10行，需要9条水平线
            y = regions[row * 9]["center"][1]  # 每行的中心y坐标
            cv2.line(visualized_image, (0, int(y)), (width, int(y)), (0, 255, 0), 1)
        
        # 绘制垂直线（列分隔线）
        for col in range(1, 9):  # 9列，需要8条垂直线
            x = regions[col]["center"][0]  # 每列的中心x坐标
            cv2.line(visualized_image, (int(x), 0), (int(x), height), (0, 255, 0), 1)
        
        # 绘制区域编号（可选）
        for row in range(10):
            for col in range(9):
                region_idx = row * 9 + col
                if region_idx < len(regions):
                    center_x, center_y = regions[region_idx]["center"]
                    # 绘制区域编号
                    cv2.putText(visualized_image, f"{region_idx}", 
                                (int(center_x) - 5, int(center_y) + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        return visualized_image
    
    def publish_region_matrix(self, matrix):
        """发布区域状态矩阵"""
        try:
            msg = RegionMatrix()
            msg.header.stamp = rospy.Time.now()
            msg.data = '\n'.join([" ".join(map(str, row)) for row in matrix])
            self.board_pub.publish(msg)
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
                regions = self.calculate_grid_size()
                
                # 检测棋子
                detections = self.detect_pieces(cv_image)
                
                # 生成区域矩阵
                region_matrix = self.generate_region_matrix(detections, regions)
                final_matrix = self.replace_spaces_with_zero(region_matrix)
                
                # 可视化网格和棋子
                visualized_image = self.visualize_grid(cv_image.copy(), regions)
                
                # 显示可视化图像（可选，用于调试）
                rospy.loginfo(f"生成矩阵:\n{final_matrix}")
                cv2.imshow("Chessboard Detection", visualized_image)
                cv2.waitKey(1)
                
                # 发布结果
                self.publish_region_matrix(final_matrix)

            except Exception as e:
                rospy.logerr(f"运行时错误: {str(e)}")
                
            self.rate.sleep()

if __name__ == '__main__':
    try:
        detector = ChessboardDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down chessboard detector node")