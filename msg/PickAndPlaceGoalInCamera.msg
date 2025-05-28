# ee368_project/msg/PickAndPlaceGoalInCamera.msg
string object_id_at_pick    # 可选，抓取对象的ID
geometry_msgs/Point pick_position_in_camera   # 抓取点在相机坐标系中的位置 (x,y,z)

string target_location_id_at_place # 可选，放置位置的ID
geometry_msgs/Point place_position_in_camera  # 放置点在相机坐标系中的位置 (x,y,z)