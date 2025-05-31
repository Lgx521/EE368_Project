#!/usr/bin/env python3

import rospy
import subprocess
from std_msgs.msg import String # 或者你用来触发的消息类型

class NodeKiller:
    def __init__(self):
        # 节点初始化只应调用一次
        # 使用 anonymous=True 确保节点名称唯一，如果需要固定名称则设为 False
        rospy.init_node('node_killer_controller', anonymous=True)

        # 从参数服务器获取参数，如果未设置则使用默认值
        # 确保在 launch 文件中或通过命令行正确设置这些参数
        # target_node_to_kill 应该是目标节点的完整 ROS 名称，例如 /my_target_node
        self.target_node_name = "/chessboard_detector_node"
        
        default_trigger_topic = "/kill_trigger"
        self.trigger_topic = rospy.get_param('~trigger_topic', default_trigger_topic)

        self.killed_once = False # 标志位，确保只尝试杀死一次

        # 订阅触发主题
        self.sub = rospy.Subscriber(self.trigger_topic, String, self.trigger_callback, queue_size=1)
        # queue_size=1 确保我们只处理最新的消息，如果快速连续发送多个kill信号

        rospy.loginfo(f"Controller node '{rospy.get_name()}' initialized.")
        rospy.loginfo(f"Listening on topic '{self.trigger_topic}' to kill node '{self.target_node_name}'.")

    def trigger_callback(self, msg: String): # 使用类型提示
        rospy.loginfo(f"Received trigger message on '{self.trigger_topic}': '{msg.data}'")

        if self.killed_once:
            rospy.loginfo(f"Kill command for node '{self.target_node_name}' already attempted. Ignoring.")
            return

        # 确保目标节点名称是完整的全局名称 (以 '/' 开头)
        node_to_kill_final = self.target_node_name
        if not node_to_kill_final.startswith('/'):
            rospy.logwarn(f"Target node name '{self.target_node_name}' does not start with '/'. Prepending '/' to form '{'/' + self.target_node_name}'.")
            node_to_kill_final = '/' + node_to_kill_final
        
        rospy.loginfo(f"Attempting to kill node: '{node_to_kill_final}'")

        try:
            # 构建 rosnode kill 命令
            command = ["rosnode", "kill", node_to_kill_final]
            rospy.loginfo(f"Executing command: \"{' '.join(command)}\"")

            # 执行命令
            # 使用 Popen 以便获取 stdout 和 stderr
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # text=True (Python 3.7+) 会自动解码 stdout/stderr 为字符串
            # 如果是 Python 3.6 或更早，需要手动 .decode()
            # stdout_bytes, stderr_bytes = process.communicate()
            # stdout_str = stdout_bytes.decode().strip()
            # stderr_str = stderr_bytes.decode().strip()
            
            stdout_str, stderr_str = process.communicate() # 等待命令完成并获取输出
            return_code = process.returncode

            # 检查命令执行结果
            rospy.loginfo(f"Command 'rosnode kill {node_to_kill_final}' finished with return code: {return_code}")
            if stdout_str:
                rospy.loginfo(f"rosnode kill stdout:\n{stdout_str.strip()}")
            if stderr_str:
                rospy.logwarn(f"rosnode kill stderr:\n{stderr_str.strip()}") # stderr 通常是警告或错误

            if return_code == 0:
                # `rosnode kill <non_existent_node>` 也会返回 0 并打印到 stderr: "ERROR: unknown node [...]"
                # 所以我们需要更智能地判断是否真的成功了
                if "killed" in stdout_str.lower() or ("killing" in stdout_str.lower() and "ERROR: unknown node" not in stderr_str):
                    rospy.loginfo(f"Successfully sent kill signal to node '{node_to_kill_final}'. Node should be shutting down.")
                    self.killed_once = True
                    self.sub.unregister() # 停止订阅，避免重复触发
                    rospy.loginfo(f"Unsubscribed from '{self.trigger_topic}'.")
                    # 可选: 如果控制节点任务完成，也可以让它自己关闭
                    # rospy.signal_shutdown(f"Target node '{node_to_kill_final}' killed, controller shutting down.")
                elif "ERROR: unknown node" in stderr_str:
                    rospy.logwarn(f"Node '{node_to_kill_final}' was not found by 'rosnode kill'. It might have already been shut down or the name is incorrect.")
                    self.killed_once = True # 标记为已尝试，避免对不存在的节点重复操作
                else:
                    rospy.logwarn(f"'rosnode kill {node_to_kill_final}' executed with return code 0, but output is ambiguous. Please check target node status.")
                    # 根据具体情况决定是否标记为 killed_once
                    self.killed_once = True # 通常还是标记为已尝试
            else:
                rospy.logerr(f"Failed to execute 'rosnode kill {node_to_kill_final}'. Return code: {return_code}.")
                # 此时 self.killed_once 可以保持 False 以允许重试，或者设为 True
        
        except FileNotFoundError:
            rospy.logfatal("'rosnode' command not found. Is the ROS environment (Noetic) sourced correctly in the terminal where this Python script is running? This script cannot function without 'rosnode'.")
            # 严重错误，可能需要关闭自身
            rospy.signal_shutdown("'rosnode' command not found.")
        except Exception as e:
            rospy.logerr(f"An unexpected exception occurred while trying to kill node '{node_to_kill_final}': {e}")
            # 根据具体情况决定是否标记为 killed_once

if __name__ == '__main__':
    try:
        node_killer_instance = NodeKiller()
        rospy.spin() # 保持节点运行，等待消息
    except rospy.ROSInterruptException:
        rospy.loginfo("Node Killer Controller shutting down due to ROS interrupt (e.g., Ctrl+C).")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in Node Killer Controller main: {e}")