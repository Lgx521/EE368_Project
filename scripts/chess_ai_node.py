#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ee368_project.msg import PickAndPlaceGoalInCamera

from elephant_fish import ai_move_from_matrix  

class ChessAINode:
    def __init__(self):
        rospy.init_node('chess_ai_node')

        self.current_board = None
        self.last_board = None
        self.is_ai_turn = False

        self.board_sub = rospy.Subscriber("/chess_board_matrix", String, self.board_callback)
        self.coord_pub = rospy.Publisher("/kinova_pick_place/goal_in_camera", PickAndPlaceGoalInCamera, queue_size=10)

        self.top_left = Point(x=-0.17,y=0.07,z=0.63)
        self.top_right = Point(x=-0.18,y=-0.13,z=0.63)
        self.bottom_left = Point(x=0.18,y=0.06,z=0.67)
        self.bottom_right = Point(x=0.16,y=-0.15,z=0.65)

        self.rate = rospy.Rate(10)

    def board_callback(self, msg):
        try:
            board = np.array(eval(msg.data), dtype=object)
            if self.current_board is None or not np.array_equal(board, self.current_board):
                self.current_board = board
                rospy.loginfo("Received updated board.")
        except Exception as e:
            rospy.logerr("Error parsing board message: %s", str(e))

    def matrix_to_point(self, x_value, y_value):
        x = self.bottom_left.x + x_value/8*(self.bottom_right.x-self.bottom_left.x) + y_value/9*(self.top_left.x-self.bottom_left.x)
        y = self.bottom_left.y + x_value/8*(self.bottom_right.y-self.bottom_left.y) + y_value/9*(self.top_left.y-self.bottom_left.y)
        z = self.bottom_left.z + x_value/8*(self.bottom_right.z-self.bottom_left.z) + y_value/9*(self.top_left.z-self.bottom_left.z)
        return Point(x=x, y=y, z=z)



    def run(self):
        while not rospy.is_shutdown():
            if self.current_board is None:
                self.rate.sleep()
                continue

            if self.is_ai_turn:
                rospy.loginfo("AI thinking...")
                ai_board,start,end = ai_move_from_matrix(self.current_board)

                if ai_board is not None:
                    
                    if start and end:
                        msg = PickAndPlaceGoalInCamera()
                        msg.object_id_at_pick = ""
                        msg.pick_position_in_camera = self.matrix_to_point(start[0],start[1])
                        msg.target_location_id_at_place = ""
                        msg.place_position_in_camera = self.matrix_to_point(end[0],end[1])

                        self.coord_pub.publish(msg)
                        rospy.loginfo(f"AI move published: from {self.matrix_to_point(start[0],start[1])} to {self.matrix_to_point(end[0],end[1])}")

                        self.current_board = ai_board.copy()  # 更新当前棋盘状态
                        self.last_board = ai_board.copy()
                        self.is_ai_turn = False
                    else:
                        rospy.logwarn("AI move could not be detected.")
                else:
                    rospy.logwarn("AI returned no new board.")

                self.rate.sleep()
                continue

            if self.last_board is None or not np.array_equal(self.current_board, self.last_board):
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
