#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

# --- ROS 2 Bag Reader Import ---
import sqlite3
import glob
from rclpy.serialization import deserialize_message
# -------------------------------

from youbot_kdl_utils import YoubotKinematicKDL
import itertools

class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        super().__init__('youbot_traj_cw2')
        self.kdl_youbot = YoubotKinematicKDL(self)

        # 1. Trajectory Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/EffortJointInterface_trajectory_controller/command',
            5
        )

        # 2. Marker Publisher 
        # Topic: visualization_marker (Matches default RViz config you provided)
        marker_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, 
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.checkpoint_pub = self.create_publisher(Marker, 'visualization_marker', marker_qos)
        
        # 3. Joint State Publisher
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        self._q_path = None
        self._path_index = 0
        self._publish_timer = None
        
        # Cache for markers
        self._checkpoint_markers = [] 
        self._checkpoint_positions = []
        self._checkpoint_reached = []
        self._marker_timer = None

    def run(self):
        """Runs the main coursework logic."""
        self.get_logger().info('Waiting 2 seconds for everything to load up.')
        time.sleep(2.0)
        traj, q_path = self.q2()
        self._q_path = q_path
        self._path_index = 0
        self.get_logger().info('Markers published. Starting movement in 1 second...')
        time.sleep(1.0)
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)
        self._publish_timer = self.create_timer(0.05, self._publish_next_state)
        self._marker_timer = self.create_timer(1.0, self._republish_markers)

    def q2(self):
        ############################################################################
        # QUESTION E START
        ############################################################################

        init_q = np.array(self.kdl_youbot.kdl_jnt_array_to_list(
            self.kdl_youbot.current_joint_position
        ), dtype=float)

        target_q, target_tfs = self.load_targets()  # target_q:(5,4), target_tfs:(4,4,4)

        start_tf = self.kdl_youbot.forward_kinematics(init_q)
        checkpoints_tf = np.concatenate((start_tf[:, :, None], target_tfs), axis=2)  # (4,4,5)

        self._checkpoint_positions = [checkpoints_tf[:3, 3, i] for i in range(checkpoints_tf.shape[2])]
        self._checkpoint_reached = [False] * checkpoints_tf.shape[2]
        self.init_markers(checkpoints_tf)

        # shortest visiting sequence
        order = self.get_shortest_path(checkpoints_tf)

        num_points = 8
        full_tfs = self.intermediate_tfs(order, checkpoints_tf, num_points)  # (4,4,M)

        # TF -> joints
        q_path = self.full_checkpoints_to_joints(full_tfs, init_q)  # (5,M)

        # JointTrajectory
        traj = JointTrajectory()
        traj.points = []

        dt = 0.05
        for i in range(q_path.shape[1]):
            pt = JointTrajectoryPoint()
            pt.positions = q_path[:, i].tolist()

            t = i * dt
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nanosec)

            traj.points.append(pt)

        return traj, q_path
        ############################################################################
        # QUESTION E END
        ############################################################################

    def load_targets(self):
        """
        Loads the target joint positions from the bagfile.
        """
        ############################################################################
        # QUESTION A START
        ############################################################################
        import os

        pkg = get_package_share_directory("cw2q2")
        bag_dir = os.path.join(pkg, "bags", "data_ros2")
        db_path = glob.glob(os.path.join(bag_dir, "*.db3"))[0]

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM topics WHERE name=?", ("joint_data",))
        topic_id = cur.fetchone()[0]

        cur.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp ASC", (topic_id,))
        rows = cur.fetchall()
        conn.close()

        qs, tfs = [], []
        for (blob,) in rows:
            msg = deserialize_message(blob, JointState)

            if len(msg.position) >= 5:
                q = np.array(msg.position[:5], dtype=float)
                tfs.append(self.kdl_youbot.forward_kinematics(q))
                qs.append(q)
            else:
                raise RuntimeError("JointState position has less than 5 values.")


        if len(qs) != 4:
            raise RuntimeError(f"Expected 4 targets, got {len(qs)}")

        target_q = np.array(qs).T
        target_tfs = np.stack(tfs, axis=2)
        self._checkpoint_positions = [target_tfs[:3, 3, i] for i in range(target_tfs.shape[2])]
        self._checkpoint_reached = [False] * target_tfs.shape[2]

        return target_q, target_tfs
        ############################################################################
        # QUESTION A END
        ############################################################################

    def get_shortest_path(self, checkpoints_tf):
        """
        Computes the order of checkpoints.
        """
        ############################################################################
        # QUESTION B START
        ############################################################################
        N = checkpoints_tf.shape[2]
        pts = checkpoints_tf[:3, 3, :].T   # (N,3)

        start = 0
        others = [i for i in range(N) if i != start]

        best_order = None
        best_dist = float("inf")

        for perm in itertools.permutations(others):
            order = [start] + list(perm)
            d = 0.0
            for i in range(len(order) - 1):
                d += np.linalg.norm(pts[order[i+1]] - pts[order[i]])
            if d < best_dist:
                best_dist = d
                best_order = order

        self.get_logger().info(f"Best order: {best_order}, total dist={best_dist:.4f}")
        return np.array(best_order, dtype=int)

        ############################################################################
        # QUESTION B END
        ############################################################################

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """
        Create intermediate transformations.
        """
        ############################################################################
        # QUESTION C START
        ############################################################################

        idx = list(sorted_checkpoint_idx)
        full = [target_checkpoint_tfs[:, :, idx[0]]]

        for i in range(len(idx) - 1):
            Ta = target_checkpoint_tfs[:, :, idx[i]]
            Tb = target_checkpoint_tfs[:, :, idx[i + 1]]
            seg = self.decoupled_rot_and_trans(Ta, Tb, num_points)


            for k in range(seg.shape[2]):
                full.append(seg[:, :, k])

        return np.stack(full, axis=2)
        ############################################################################
        # QUESTION C END
        ############################################################################

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """
        Interpolate between two transforms.
        """
        ############################################################################
        # QUESTION C START
        ############################################################################

        Ta, Tb = checkpoint_a_tf, checkpoint_b_tf
        Ra, Rb = Ta[:3, :3], Tb[:3, :3]
        pa, pb = Ta[:3, 3], Tb[:3, 3]

        def skew(w):
            return np.array([[0, -w[2], w[1]],
                            [w[2], 0, -w[0]],
                            [-w[1], w[0], 0]], dtype=float)

        Rrel = Ra.T @ Rb
        c = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(c)

        if theta < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = np.array([Rrel[2, 1] - Rrel[1, 2],
                            Rrel[0, 2] - Rrel[2, 0],
                            Rrel[1, 0] - Rrel[0, 1]], dtype=float) / (2.0 * np.sin(theta))

        tfs = []

        # 1
        K = skew(axis)
        K2 = K @ K
        I = np.eye(3)
        for k in range(1, num_points + 1):
            a = k / num_points
            Rinc = I + np.sin(a * theta) * K + (1 - np.cos(a * theta)) * K2
            T = np.eye(4)
            T[:3, :3] = Ra @ Rinc
            T[:3, 3] = pa
            tfs.append(T)

        # 2
        for k in range(1, num_points + 1):
            a = k / num_points
            T = np.eye(4)
            T[:3, :3] = Rb
            T[:3, 3] = pa + a * (pb - pa)
            tfs.append(T)

        return np.stack(tfs, axis=2)
        ############################################################################
        # QUESTION C END
        ############################################################################

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """
        Compute associated joint positions.
        """
        ############################################################################
        # QUESTION D START
        ############################################################################
        M = full_checkpoint_tfs.shape[2]
        q_path = np.zeros((5, M), dtype=float)

        q = np.array(init_joint_position, dtype=float).copy()

        for i in range(M):
            T_des = full_checkpoint_tfs[:, :, i]
            q = self.ik_position_only(T_des, q)
            q_path[:, i] = q

        return q_path
        ############################################################################
        # QUESTION D END
        ############################################################################

    def ik_position_only(self, pose, q0):
        """
        Iterative Inverse Kinematics.
        """
        ############################################################################
        # QUESTION D START
        ############################################################################
        p_des = pose[:3, 3]
        q = np.array(q0, dtype=float).copy()

        max_iters = 200
        tol = 1e-3
        lam = 0.05
        alpha = 0.6

        for _ in range(max_iters):
            T = self.kdl_youbot.forward_kinematics(q)
            p = T[:3, 3]
            err = p_des - p

            if np.linalg.norm(err) < tol:
                break

            J = self.kdl_youbot.get_jacobian(q)
            Jp = J[:3, :]

            # dq = J^T (J J^T + lam^2 I)^(-1) err
            A = Jp @ Jp.T + (lam ** 2) * np.eye(3)
            dq = Jp.T @ np.linalg.solve(A, err)

            q = q + alpha * dq

        return q
        ############################################################################
        # QUESTION D END
        ############################################################################

    def init_markers(self, tfs):
        """
        Creates markers ONLY for the 5 checkpoints.
        """
        self._checkpoint_markers = []
        marker_id = 0
        
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = marker_id
            marker_id += 1
            marker.header.frame_id = 'base_link' 
            marker.ns = "points_and_lines" 
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.scale.x = 0.04 
            marker.scale.y = 0.04
            marker.scale.z = 0.04
            
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker.lifetime = Duration(sec=0, nanosec=0)
            marker.frame_locked = True
            
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = float(tfs[0, -1, i])
            marker.pose.position.y = float(tfs[1, -1, i])
            marker.pose.position.z = float(tfs[2, -1, i])
            
            self._checkpoint_markers.append(marker)
        
        self._republish_markers()

    def _republish_markers(self):
        """Timer callback to keep markers visible and update colors."""
        if not self._checkpoint_markers:
            return
        stamp_zero = rclpy.time.Time(seconds=0).to_msg()
            
        for i, marker in enumerate(self._checkpoint_markers):
            marker.header.stamp = stamp_zero
            reached = self._checkpoint_reached[i] if i < len(self._checkpoint_reached) else False
            
            if reached:
                marker.color.r = 0.0
                marker.color.g = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
            marker.color.b = 0.0
            
            self.checkpoint_pub.publish(marker)

    def _publish_next_state(self):
        """Timer callback to visualize the robot moving."""
        if self._q_path is None:
            return
        if self._path_index >= self._q_path.shape[1]:
            self.destroy_timer(self._publish_timer)
            return
            
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        msg.position = self._q_path[:, self._path_index].tolist()
        self.joint_state_pub.publish(msg)
        
        ee_pose = self.kdl_youbot.forward_kinematics(self._q_path[:, self._path_index])
        ee_pos = ee_pose[:3, 3]
        
        for idx, cp in enumerate(self._checkpoint_positions):
            if cp is None or len(cp) == 0: continue
            if np.linalg.norm(ee_pos - cp) < 0.03: 
                self._checkpoint_reached[idx] = True
                
        self._path_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = YoubotTrajectoryPlanning()
    node.run()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
