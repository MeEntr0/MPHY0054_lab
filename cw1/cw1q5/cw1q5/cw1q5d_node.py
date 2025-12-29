#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
# Import from the other node file within the same ROS 2 package
from cw1q5.cw1q5b_node import forward_kinematics
from geometry_msgs.msg import TransformStamped, Quaternion

"""
This node subscribes to the /joint_states topic, applies the necessary polarity
and offset corrections to the joint angles, computes the forward kinematics for
each joint frame, and publishes the transformations to TF2 for visualization
in RViz.
"""


# ╔════════════════════════════════════════════════════════════════════════╗
# ║           SOLUTION FOR PART 1: DH PARAMETERS & JOINT OFFSETS           ║
# ╚════════════════════════════════════════════════════════════════════════╝
youbot_dh_parameters = {
    'a':     [0.033,   0.155, 0.135, 0.0,   -0.002],
    'alpha': [-np.pi/2, 0.0,  0.0,  -np.pi/2, 0.0],
    'd':     [0.147,   0.0,   0.0,  0.0,    0.218],
    'theta': [0.0, -np.pi/2, 0.0, -np.pi/2, -np.pi]
}
youbot_joint_offsets = [
    170.0 * np.pi / 180.0,    
    -65.0 * np.pi / 180.0,    
    146.0 * np.pi / 180.0,    
    -102.5 * np.pi / 180.0,   
    167.5 * np.pi / 180.0     
]

youbot_dh_offset_paramters = youbot_dh_parameters.copy()
youbot_dh_offset_paramters['theta'] = [
    theta + offset
    for theta, offset in zip(youbot_dh_offset_paramters['theta'], youbot_joint_offsets)
]

youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]
# ╔════════════════════════════════════════════════════════════════════════╗
# ║                        END OF SOLUTION FOR PART 1                      ║
# ╚════════════════════════════════════════════════════════════════════════╝


def rotmat2q(R):
    """Function for converting a 3x3 Rotation matrix R to quaternion q."""
    q = Quaternion()
    angle = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)

    if np.isclose(angle, 0.0):
        q.w = 1.0
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
    else:
        xr = R[2, 1] - R[1, 2]
        yr = R[0, 2] - R[2, 0]
        zr = R[1, 0] - R[0, 1]
        norm = np.sqrt(np.power(xr, 2) + np.power(yr, 2) + np.power(zr, 2))
        x = xr / norm
        y = yr / norm
        z = zr / norm
        q.w = np.cos(angle / 2)
        q.x = x * np.sin(angle / 2)
        q.y = y * np.sin(angle / 2)
        q.z = z * np.sin(angle / 2)

    return q


class ForwardKinematicsOffsetNode(Node):
    def __init__(self):
        super().__init__('forward_kinematic_offset_node')
        
        self.br = TransformBroadcaster(self)
        
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                     PART 3: INITIALIZE ROS 2 SUBSCRIBER                ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',         
            self.fkine_wrapper,      
            10                       
        )
        self.subscription
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                              END OF PART 3                             ║
        # ╚════════════════════════════════════════════════════════════════════════╝

    def fkine_wrapper(self, joint_msg):
        """
        Callback function to compute FK and publish transforms.
        """
        assert isinstance(joint_msg, JointState), "Node must subscribe to a topic where JointState messages are published"
        
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                      PART 2: FKINE WRAPPER IMPLEMENTATION              ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        joints = list(joint_msg.position[:5])

        joints_corr = [
            sign * angle
            for sign, angle in zip(youbot_joint_readings_polarity, joints)
        ]

        for i in range(1, 6):  
            T = forward_kinematics(
                youbot_dh_offset_paramters,
                joints_corr,
                up_to_joint=i
            )

            p = T[0:3, 3]
            R = T[0:3, 0:3]
            q = rotmat2q(R)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'base_link'
            tf_msg.child_frame_id = f'arm_link_{i}_offset'

            tf_msg.transform.translation.x = float(p[0])
            tf_msg.transform.translation.y = float(p[1])
            tf_msg.transform.translation.z = float(p[2])
            tf_msg.transform.rotation = q

            self.br.sendTransform(tf_msg)
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                              END OF PART 2                             ║
        # ╚════════════════════════════════════════════════════════════════════════╝


def main(args=None):
    # Standard ROS 2 main function
    rclpy.init(args=args)
    fk_offset_node = ForwardKinematicsOffsetNode()
    rclpy.spin(fk_offset_node)
    
    # Destroy the node explicitly
    fk_offset_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
