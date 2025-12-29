#!/usr/bin/env python3

import numpy as np
from cw1q9.youbotKineBase import YoubotKinematicBase
import rclpy

class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super().__init__('youbot_kinematic_student', tf_suffix='student')
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                   FILL IN the JOINT OFFSETS FOUND IN CW1Q5             ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        youbot_joint_offsets = [
            170.0 * np.pi / 180.0,    
            -65.0 * np.pi / 180.0,    
            146.0 * np.pi / 180.0,    
            -102.5 * np.pi / 180.0,   
            167.5 * np.pi / 180.0     
        ]

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ╚════════════════════════════════════════════════════════════════════════╝
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        T = np.identity(4)
        
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)
            
        return T

    def get_jacobian(self, joint):
        """Given the joint values of the robot, compute the Jacobian matrix. Coursework 1 Question 9a.
        Reference - Lecture 5 slide 24.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        assert isinstance(joint, list)
        assert len(joint) == 5

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                  YOUR CODE STARTS HERE: CALCULATE JACOBIAN             ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        joint = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joint)]

        T = np.identity(4)
        origins = [T[:3, 3]]
        z_axes = [np.array([0, 0, -1])]  

        for i in range(5):
            A = self.standard_dh(self.dh_params["a"][i],
                                self.dh_params["alpha"][i],
                                self.dh_params["d"][i],
                                self.dh_params["theta"][i] + joint[i])
            T = T.dot(A)
            origins.append(T[:3, 3])
            z_axes.append(T[:3, 2])      
        

        jacobian = np.zeros((6, 5))
        o_end = origins[-1]

        for i in range(5):
            zi = z_axes[i]
            oi = origins[i]
            jacobian[0:3, i] = np.cross(zi, o_end - oi)  
            jacobian[3:6, i] = zi                         
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ╚════════════════════════════════════════════════════════════════════════╝
        assert jacobian.shape == (6, 5)
        return jacobian

    def check_singularity(self, joint):
        """Check for singularity condition given robot joints. Coursework 1 Question 9c.
        Reference Lecture 5 slide 30.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(joint, list)
        assert len(joint) == 5
        
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║                  YOUR CODE STARTS HERE: CHECK SINGULARITY              ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        J = self.get_jacobian(joint)

        Jv = J[0:3, :]

        rank = np.linalg.matrix_rank(Jv, tol=1e-4)

        if rank < 3:
            singularity = True
        else:
            singularity = False
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ╚════════════════════════════════════════════════════════════════════════╝
        assert isinstance(singularity, bool)
        return singularity

def main(args=None):
    rclpy.init(args=args)
    node = YoubotKinematicStudent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()