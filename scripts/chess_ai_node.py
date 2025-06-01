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

from elephant_fish import ai_move_from_matrix  

class ChessAINode:
    def __init__(self):
        rospy.init_node('chess_ai_node')
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


        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        self.garbage_point = None


        self.rate = rospy.Rate(10)

    def corner_callback(self,msg):
        self.top_left = msg.top_left
        self.top_right = msg.top_right
        self.bottom_left = msg.bottom_left
        self.bottom_right = msg.bottom_right
        self.garbage_point = Point()
        self.garbage_point.x = 2 * self.bottom_left.x - self.bottom_right.x
        self.garbage_point.y = 2 * self.bottom_left.y - self.bottom_right.y
        self.garbage_point.z = 2 * self.bottom_left.z - self.bottom_right.z      
        
        # 打印接收到的坐标
        rospy.loginfo("Received chessboard corners")
        rospy.loginfo("Top Left: (%.2f, %.2f, %.2f)", self.top_left.x, self.top_left.y, self.top_left.z)
        rospy.loginfo("Top Right: (%.2f, %.2f, %.2f)", self.top_right.x, self.top_right.y, self.top_right.z)
        rospy.loginfo("Bottom Left: (%.2f, %.2f, %.2f)", self.bottom_left.x, self.bottom_left.y, self.bottom_left.z)
        rospy.loginfo("Bottom Right: (%.2f, %.2f, %.2f)", self.bottom_right.x, self.bottom_right.y, self.bottom_right.z)   

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
        z = self.bottom_left.z + x_value/8*(self.bottom_right.z-self.bottom_left.z) + y_value/9*(self.top_left.z-self.bottom_left.z)
        return Point(x=x, y=y, z=z)



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

                if ai_board is not None:
                    
                    if start and end:
                        if self.current_board[9-end[1]][end[0]] != '0':
                            rospy.loginfo("chess eating detected")
                            msg0 = PickAndPlaceGoalInCamera()
                            msg0.object_id_at_pick = ""
                            msg0.pick_position_in_camera = self.matrix_to_point(end[0],end[1])
                            msg0.target_location_id_at_place = ""
                            msg0.place_position_in_camera = self.garbage_point
                            self.coord_pub.publish(msg0)
                            rospy.sleep(20)  

                        msg1 = PickAndPlaceGoalInCamera()
                        msg1.object_id_at_pick = ""
                        msg1.pick_position_in_camera = self.matrix_to_point(start[0],start[1])
                        msg1.target_location_id_at_place = ""
                        msg1.place_position_in_camera = self.matrix_to_point(end[0],end[1])
                        self.coord_pub.publish(msg1)
                        rospy.loginfo(f"AI move published: from {self.matrix_to_point(start[0],start[1])} to {self.matrix_to_point(end[0],end[1])}")
                        rospy.loginfo(f"AI move published: from {start[0],start[1]} to {end[0],end[1]}")
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
