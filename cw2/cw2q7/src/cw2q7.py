#!/usr/bin/env python3

import os
import time
import glob
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from builtin_interfaces.msg import Duration

from cw2q4.iiwa14DynStudent import Iiwa14DynamicStudent




class CW2Q7(Node):
    def __init__(self):
        super().__init__("cw2q7_node")

        # launch 里给的参数（a/c/d 会用）
        self.declare_parameter("bag_topic", "/iiwa/EffortJointInterface_trajectory_controller/command")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("command_topic", "/iiwa_controller/joint_trajectory")

        self.bag_topic = self.get_parameter("bag_topic").value
        self.command_topic = self.get_parameter("command_topic").value

        # bag 默认路径：share/cw2q7/bag/data_ros2
        share = get_package_share_directory("cw2q7")
        self.bag_dir = os.path.join(share, "bag", "data_ros2")

        # 发布轨迹到控制器
        self.traj_pub = self.create_publisher(JointTrajectory, self.command_topic, 10)
        self.ctrl_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

        # d: 用 Q4 的学生版 KDL 动力学（get_B / get_C_times_qdot / get_G）
        self.model = Iiwa14DynamicStudent()

        # 订阅 joint_states
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.joint_sub = self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 50)

        # 记录用
        self._collecting = False
        self._t0 = None
        self.t_arr = None
        self.qdd_arr = None
        self._t_log = []
        self._qdd_log = []

    def _joint_state_cb(self, msg: JointState):
        if not self._collecting:
            return

        if len(msg.position) < 7:
            return

        # 对齐顺序
        if msg.name and hasattr(self, "ctrl_joints"):
            try:
                idx = [msg.name.index(n) for n in self.ctrl_joints]
            except ValueError:
                return
            q = np.array([msg.position[i] for i in idx], dtype=float)
            qdot = np.array([msg.velocity[i] for i in idx], dtype=float) if len(msg.velocity) >= 7 else np.zeros(7)
            tau = np.array([msg.effort[i] for i in idx], dtype=float) if len(msg.effort) >= 7 else np.zeros(7)
        else:
            q = np.array(msg.position[:7], dtype=float)
            qdot = np.array(msg.velocity[:7], dtype=float) if len(msg.velocity) >= 7 else np.zeros(7)
            tau = np.array(msg.effort[:7], dtype=float) if len(msg.effort) >= 7 else np.zeros(7)

        # student KDL那边要 list
        q_list = q.tolist()
        qd_list = qdot.tolist()

        B = np.array(self.model.get_B(q_list), dtype=float)
        Cqdot = np.array(self.model.get_C_times_qdot(q_list, qd_list), dtype=float)
        G = np.array(self.model.get_G(q_list), dtype=float)

        # forward dynamics: ddq = B^{-1}(tau - C - G)
        qdd = np.linalg.solve(B, (tau - Cqdot - G))

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = t_now
        self._t_log.append(t_now - self._t0)
        self._qdd_log.append(qdd)

    def run(self):
        self.get_logger().info("CW2Q7 start")
        time.sleep(1.0)

        self.q7a_load_bag_and_inspect()
        self.q7c_publish_trajectory()
        self.q7d_compute_joint_accelerations()
        self.q7e_plot_joint_accelerations()

    def q7a_load_bag_and_inspect(self):
        """
        a. Load the bag from the bagfile. What type of message does the bagfile contain?
           How many messages? What is the content of the messages?
        """
        db_path = glob.glob(os.path.join(self.bag_dir, "*.db3"))[0]

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # topic id + type
        topic_id, type_str = cur.execute(
            "SELECT id, type FROM topics WHERE name=?",
            (self.bag_topic,)
        ).fetchone()

        # 消息数
        n = cur.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id=?",
            (topic_id,)
        ).fetchone()[0]

        # 取第一条消息做内容示例
        _, blob = cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
            (topic_id,)
        ).fetchone()

        msg = deserialize_message(blob, JointTrajectory)

        # 只打印题目要点（简单一点）
        self.get_logger().info(f"[Q7a] message type: {type_str}")
        self.get_logger().info(f"[Q7a] message count: {n}")
        self.get_logger().info(
            f"[Q7a] content: JointTrajectory with {len(msg.joint_names)} joints and {len(msg.points)} points"
        )

        conn.close()

    def q7c_publish_trajectory(self):
        """
        c. 从bag读JointTrajectory并发布，让机器人动起来
        """
        # 1) 找db3
        db_path = glob.glob(os.path.join(self.bag_dir, "*.db3"))[0]

        # 2) 读出bag里那条JointTrajectory（本bag只有1条）
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        topic_id = cur.execute(
            "SELECT id FROM topics WHERE name=?",
            (self.bag_topic,)
        ).fetchone()[0]

        _, blob = cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
            (topic_id,)
        ).fetchone()

        traj = deserialize_message(blob, JointTrajectory)
        conn.close()

        # 3) 让控制器更容易接受：补一下stamp
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.ctrl_joints

        # 4) 如果time_from_start都为0，简单补一个递增时间
        all_zero = True
        for p in traj.points:
            if p.time_from_start.sec != 0 or p.time_from_start.nanosec != 0:
                all_zero = False
                break

        if all_zero:
            dt = 2.0  # 每个点间隔2秒，方便看
            for i, p in enumerate(traj.points):
                t = (i + 1) * dt
                sec = int(t)
                nanosec = int((t - sec) * 1e9)
                p.time_from_start = Duration(sec=sec, nanosec=nanosec)

        # 5) 稍等一下再发（给controller起起来的时间）
        time.sleep(1.0)
        self.traj_pub.publish(traj)
        self.get_logger().info(f"[Q7c] published {len(traj.points)} points")
        time.sleep(6.0)

    def q7d_compute_joint_accelerations(self):
        # 简单粗暴：收集10秒（题目也允许加delay）
        duration_sec = 10.0

        self._t_log = []
        self._qdd_log = []
        self._t0 = None

        self._collecting = True
        t_start = time.time()
        while time.time() - t_start < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._collecting = False

        # 整理成numpy，给e画图用
        if len(self._qdd_log) > 0:
            self.t_arr = np.array(self._t_log, dtype=float)           # (N,)
            self.qdd_arr = np.array(self._qdd_log, dtype=float).T     # (7,N)
            self.get_logger().info(f"[Q7d] collected {self.qdd_arr.shape[1]} samples")
        else:
            self.get_logger().info("[Q7d] no data collected (check /joint_states)")

    def q7e_plot_joint_accelerations(self):

        if self.t_arr is None or self.qdd_arr is None:
            self.get_logger().info("[Q7e] nothing to plot (run Q7d first)")
            return

        # t: (N,), qdd: (7,N)
        t = self.t_arr - self.t_arr[0]
        qdd = self.qdd_arr

        # 输出目录：~/.ros/cw2q7
        out_dir = os.path.join(os.path.expanduser("~"), ".ros", "cw2q7")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "q7e_joint_accel.png")

        plt.figure()
        for i in range(7):
            plt.plot(t, qdd[i, :], label=f"joint_{i+1}")

        plt.xlabel("time [s]")
        plt.ylabel("joint acceleration [rad/s^2]")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        self.get_logger().info(f"[Q7e] saved plot to {out_path}")



def main(args=None):
    rclpy.init(args=args)
    node = CW2Q7()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

