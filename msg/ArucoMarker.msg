# Represents a single detected ArUco marker
std_msgs/Header header   # ROS Header (stamp, frame_id)
int32 id                 # Marker ID
geometry_msgs/Pose pose_camera  # Pose of the marker in camera frame
                         # pose.position: (x,y,z) in meters
                         # pose.orientation: (x,y,z,w) quaternion
# float64 confidence       # Optional: a confidence score for the detection (not used in this basic example)