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

        # arg
        self.declare_parameter("bag_topic", "/iiwa/EffortJointInterface_trajectory_controller/command")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("command_topic", "/iiwa_controller/joint_trajectory")

        self.bag_topic = self.get_parameter("bag_topic").value
        self.command_topic = self.get_parameter("command_topic").value

        share = get_package_share_directory("cw2q7")
        self.bag_dir = os.path.join(share, "bag", "data_ros2")

        self.traj_pub = self.create_publisher(JointTrajectory, self.command_topic, 10)
        self.ctrl_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

        # Q4 
        self.model = Iiwa14DynamicStudent()

        # sub joint_states
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.joint_sub = self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 50)

        # record
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

        # align
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

        q_list = q.tolist()
        qd_list = qdot.tolist()

        B = np.array(self.model.get_B(q_list), dtype=float)
        Cqdot = np.array(self.model.get_C_times_qdot(q_list, qd_list), dtype=float)
        G = np.array(self.model.get_G(q_list), dtype=float)

        # forward dynamics
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
        db_path = glob.glob(os.path.join(self.bag_dir, "*.db3"))[0]

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # topic id + type
        topic_id, type_str = cur.execute(
            "SELECT id, type FROM topics WHERE name=?",
            (self.bag_topic,)
        ).fetchone()

        # message
        n = cur.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id=?",
            (topic_id,)
        ).fetchone()[0]

        # first message
        _, blob = cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
            (topic_id,)
        ).fetchone()

        msg = deserialize_message(blob, JointTrajectory)

        self.get_logger().info(f"[Q7a] message type: {type_str}")
        self.get_logger().info(f"[Q7a] message count: {n}")
        self.get_logger().info(
            f"[Q7a] content: JointTrajectory with {len(msg.joint_names)} joints and {len(msg.points)} points"
        )

        conn.close()

    def q7c_publish_trajectory(self):
        db_path = glob.glob(os.path.join(self.bag_dir, "*.db3"))[0]

        # JointTrajectory in bag
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

        # stamp
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.ctrl_joints

        all_zero = True
        for p in traj.points:
            if p.time_from_start.sec != 0 or p.time_from_start.nanosec != 0:
                all_zero = False
                break

        if all_zero:
            dt = 2.0  
            for i, p in enumerate(traj.points):
                t = (i + 1) * dt
                sec = int(t)
                nanosec = int((t - sec) * 1e9)
                p.time_from_start = Duration(sec=sec, nanosec=nanosec)

        # wait
        time.sleep(1.0)
        self.traj_pub.publish(traj)
        self.get_logger().info(f"[Q7c] published {len(traj.points)} points")
        time.sleep(6.0)

    def q7d_compute_joint_accelerations(self):
        # collect 10s
        duration_sec = 10.0

        self._t_log = []
        self._qdd_log = []
        self._t0 = None

        self._collecting = True
        t_start = time.time()
        while time.time() - t_start < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._collecting = False

        # for e
        if len(self._qdd_log) > 0:
            self.t_arr = np.array(self._t_log, dtype=float)  
            self.qdd_arr = np.array(self._qdd_log, dtype=float).T
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

