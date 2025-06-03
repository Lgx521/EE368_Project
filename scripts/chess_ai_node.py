#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ee368_project.msg import PickAndPlaceGoalInCamera
from ee368_project.msg import ChessboardCorners

from ee368_project.msg import RegionMatrix

from elephant_fish import ai_move_from_matrix  

class ChessAINode:
    def __init__(self):
        rospy.init_node('chess_ai_node')
        
        self.init_board = rospy.Subscriber('/chess_board_matrix', RegionMatrix, self.matrix_callback)
        self.init_board = np.array([['r','n','b','a','k','a','b','n','r'], 
        [0,0,0,0,0,0,0,0,0], 
        [0,'c',0,0,0,0,0,'c',0], ['p',0,'p',0,'p',0,'p',0,'p'], 
        [0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,0,0,0], 
        ['P',0,'P',0,'P',0,'P',0,'P'], 
        [0,'C',0,0,0,0,0,'C',0], 
        [0,0,0,0,0,0,0,0,0], ['R','N','B','A','K','A','B','N','R']])
        self.current_board = self.init_board       
        self.last_board = None
        self.is_ai_turn = False

        self.board_sub = rospy.Subscriber("/chess_board_matrix", String, self.board_callback)
        self.corner_sub = rospy.Subscriber("/chessboard_corners", ChessboardCorners, self.corner_callback)
        self.coord_pub = rospy.Publisher("/kinova_pick_place/goal_in_camera", PickAndPlaceGoalInCamera, queue_size=10)
        self.arm_status_sub = rospy.Subscriber("/my_gen3_lite/arm_status", String, self.arm_status_callback)
        self.eat_status_pub = rospy.Publisher("/ai_eat_status", String, queue_size=1, latch=True)

        self.arm_status = None
        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        self.garbage_point = None
        self.distance = None
        self.eat_turn = 0
        self.garbage_place = None

        self.rate = rospy.Rate(10)

    def arm_status_callback(self, msg: String):
        rospy.loginfo(f"接收到机械臂状态: {msg.data}")
        self.arm_status = msg.data

    def corner_callback(self,msg):
        self.top_left = msg.top_left
        self.top_right = msg.top_right
        self.bottom_left = msg.bottom_left
        self.bottom_right = msg.bottom_right
        self.garbage_point = Point()
        self.garbage_point.x = 2 * self.top_right.x - self.top_left.x
        self.garbage_point.y = 2 * self.top_right.y - self.top_left.y  
        self.garbage_point.z = 2 * self.top_right.z - self.top_left.z  
        self.distance = Point()
        self.distance.x = (self.top_right.x - self.bottom_right.x)/9*self.eat_turn
        self.distance.y = (self.top_right.y - self.bottom_right.y)/9*self.eat_turn
        self.distance.z = (self.top_right.z - self.bottom_right.z)/9*self.eat_turn
        self.garbage_place = Point()
        self.garbage_place.x = self.garbage_point.x - self.distance.x
        self.garbage_place.y = self.garbage_point.y - self.distance.y
        self.garbage_place.z = self.garbage_point.z - self.distance.z

                     
        # 打印接收到的坐标
        rospy.loginfo("Received chessboard corners")
        rospy.loginfo("Top Left: (%.2f, %.2f, %.2f)", self.top_left.x, self.top_left.y, self.top_left.z)
        rospy.loginfo("Top Right: (%.2f, %.2f, %.2f)", self.top_right.x, self.top_right.y, self.top_right.z)
        rospy.loginfo("Bottom Left: (%.2f, %.2f, %.2f)", self.bottom_left.x, self.bottom_left.y, self.bottom_left.z)
        rospy.loginfo("Bottom Right: (%.2f, %.2f, %.2f)", self.bottom_right.x, self.bottom_right.y, self.bottom_right.z)
        self.corner_sub.unregister()
        rospy.loginfo("Unsubscribed from /chessboard_corners after receiving first message.")
  
    def matrix_callback(self, msg):
        """
        回调函数：将RegionMatrix消息转换为NumPy数组
        """
        try:
            # 获取原始矩阵字符串（不是扁平化的数组）
            matrix_str = msg.data
            print(matrix_str)
            
            # 调用修改后的函数（支持原始矩阵字符串）
            self.current_board = self.string_to_numpy_matrix(matrix_str)
            print(self.current_board)
            
            rospy.loginfo(f"Board shape: {self.current_board.shape}")
            
        except Exception as e:
            rospy.logerr(f"Error parsing board message: {str(e)}")

    def string_to_numpy_matrix(self, matrix_str):
        # Step 1: Split into lines and clean each line
        lines = matrix_str.splitlines()

        cleaned_lines = []
        
        for line in lines:
            # Remove extra spaces and split into elements
            elements = [elem.strip() for elem in line.split()]
            cleaned_lines.append(elements)
        
        # Step 2: Ensure consistent dimensions (10 rows x 9 columns)
        # Pad or truncate as needed
        standard_rows = 10
        standard_cols = 9
        
        # Initialize empty matrix
        matrix = []
        
        for i in range(standard_rows):
            if i < len(cleaned_lines):
                current_row = cleaned_lines[i]
                # Pad or truncate the row to 9 elements
                if len(current_row) < standard_cols:
                    # Pad with empty strings if row is too short
                    padded_row = current_row + [''] * (standard_cols - len(current_row))
                elif len(current_row) > standard_cols:
                    # Truncate if row is too long
                    padded_row = current_row[:standard_cols]
                else:
                    padded_row = current_row
                matrix.append(padded_row)
            else:
                # Add empty row if we have fewer lines than standard_rows
                matrix.append([''] * standard_cols)
        
        # Step 3: Convert to NumPy array
        np_matrix = np.array(matrix, dtype=object)
        
        return np_matrix

    def board_callback(self, msg):
        try:
            board = np.array(eval(msg.data), dtype=object)
            if self.current_board is None or not np.array_equal(board, self.current_board):
                self.current_board = board
                rospy.loginfo("Received updated board.")
                rospy.loginfo("Board equals init? %s", np.array_equal(self.current_board, self.init_board))
                rospy.loginfo("Received board: %s", self.current_board.tolist())
                rospy.loginfo("Init board: %s", self.init_board.tolist())

        except Exception as e:
            rospy.logerr("Error parsing board message: %s", str(e))

    def matrix_to_point(self, x_value, y_value):
        x = self.bottom_left.x + x_value/8*(self.bottom_right.x-self.bottom_left.x) + y_value/9*(self.top_left.x-self.bottom_left.x)
        y = self.bottom_left.y + x_value/8*(self.bottom_right.y-self.bottom_left.y) + y_value/9*(self.top_left.y-self.bottom_left.y)
        # z = self.bottom_left.z + x_value/8*(self.bottom_right.z-self.bottom_left.z) + y_value/9*(self.top_left.z-self.bottom_left.z)
        return Point(x=x+0.015, y=y, z=0.38)


    def run(self):
        while not rospy.is_shutdown():
            if not all([self.top_left, self.top_right, self.bottom_left, self.bottom_right]):
                rospy.logwarn("Waiting for chessboard corners...")
                self.rate.sleep()
                continue

            if self.current_board is None:
                self.rate.sleep()
                continue

            if self.is_ai_turn:
                rospy.loginfo("AI thinking...")
                ai_board,start,end = ai_move_from_matrix(self.current_board)

                if ai_board is not None and self.arm_status is not None:
                    
                    if start and end:
                        if self.current_board[9-end[1]][end[0]] != '0':
                            rospy.sleep(0.5)
                            if self.arm_status != '0':
                                self.rate.sleep()
                                continue
                            rospy.loginfo("chess eating detected")
                            self.eat_turn += 1
                            msg0 = PickAndPlaceGoalInCamera()
                            msg0.object_id_at_pick = ""
                            msg0.pick_position_in_camera = self.matrix_to_point(end[0],end[1])
                            msg0.target_location_id_at_place = ""
                            msg0.place_position_in_camera = self.garbage_place
                            self.coord_pub.publish(msg0)
                            self.eat_status_pub.publish(String("BUSY"))
                            rospy.loginfo("发布吃子状态: BUSY")

                        rospy.sleep(0.5)
                        if self.arm_status != '0':
                            self.rate.sleep()
                            continue  
                        msg1 = PickAndPlaceGoalInCamera()
                        msg1.object_id_at_pick = ""
                        msg1.pick_position_in_camera = self.matrix_to_point(start[0],start[1])
                        msg1.target_location_id_at_place = ""
                        msg1.place_position_in_camera = self.matrix_to_point(end[0],end[1])                                           
                        self.coord_pub.publish(msg1)
                        rospy.loginfo(f"AI move published: from {self.matrix_to_point(start[0],start[1])} to {self.matrix_to_point(end[0],end[1])}")
                        rospy.loginfo(f"AI move published: from {start[0],start[1]} to {end[0],end[1]}")
                        self.eat_status_pub.publish(String("IDLE"))
                        rospy.loginfo("发布吃子状态: IDLE")
                        self.current_board = ai_board.copy()  # 更新当前棋盘状态
                        self.last_board = ai_board.copy()
                        self.is_ai_turn = False

                    else:
                        rospy.logwarn("AI move could not be detected.")
                else:
                    rospy.logwarn("AI returned no new board.")

                self.rate.sleep()
                continue

            if (self.last_board is None or not np.array_equal(self.current_board, self.last_board)) and not np.array_equal(self.current_board,self.init_board):
                rospy.sleep(0.5)  # optional debounce
                rospy.loginfo("User move detected. Switching to AI.")
                self.is_ai_turn = True
                self.last_board = self.current_board.copy()

            self.rate.sleep()

if __name__ == "__main__":
    try:
        node = ChessAINode()
        node.run()
    except rospy.ROSInterruptException:
        pass