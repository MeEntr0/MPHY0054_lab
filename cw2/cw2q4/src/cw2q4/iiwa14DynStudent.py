#!/usr/bin/env python3

import numpy as np
import PyKDL
from ament_index_python.packages import get_package_share_directory
from cw2q4.iiwa14DynBase import Iiwa14DynamicBase
from cw2q4.urdf_kdl_utils import build_kdl_chain_from_urdf


class Iiwa14DynamicStudent(Iiwa14DynamicBase):
    def __init__(self):
        super(Iiwa14DynamicStudent, self).__init__(tf_suffix='student')
        urdf_path = get_package_share_directory('cw2q4') + '/model.urdf'
        with open(urdf_path, 'r', encoding='utf-8') as f:
            robot_description = f.read()
        self.kine_chain = build_kdl_chain_from_urdf(robot_description, "iiwa_link_0", "iiwa_link_ee")
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.jac_calc = PyKDL.ChainJntToJacSolver(self.kine_chain)
        self.dyn_solver = PyKDL.ChainDynParam(self.kine_chain, PyKDL.Vector(0, 0, -self.g))

    def forward_kinematics(self, joints_readings, up_to_joint=7):
        """Compute forward kinematics up to the selected joint."""
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        ############################################################################
        # QUESTION 4 START: implement FK only inside this file
        ############################################################################
        T = np.identity(4)
        T[2, 3] = 0.1575  # base offset

        for i in range(0, up_to_joint):
            T = T.dot(self.T_rotationZ(joints_readings[i]))
            T = T.dot(self.T_translation(self.translation_vec[i, :]))
            T = T.dot(self.T_rotationX(self.X_alpha[i]))
            T = T.dot(self.T_rotationY(self.Y_alpha[i]))

        return T
        ############################################################################
        # QUESTION 4 END
        ############################################################################

    def get_jacobian_centre_of_mass(self, joint_readings, up_to_joint=7):
        """Compute the Jacobian matrix at the centre of mass."""
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        ############################################################################
        jacobian = np.zeros((6, 7))

        # 1) 先得到“第 up_to_joint 个 link 的质心”在 base 下的位置 p_com
        T_com = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint)
        p_com = T_com[0:3, 3]

        # 2) 对每个会影响该 link 的关节 i 计算 Jacobian 列
        #    i >= up_to_joint 的关节不会影响这个 link 的 CoM，列保持 0
        for i in range(up_to_joint):
            T_i = self.forward_kinematics(joint_readings, i)
            p_i = T_i[0:3, 3]
            z_i = T_i[0:3, 0:3].dot(np.array([0, 0, 1]))

            jacobian[0:3, i] = np.cross(z_i, (p_com - p_i))  # 线速度部分
            jacobian[3:6, i] = z_i                           # 角速度部分（转动关节）

        return jacobian
        ############################################################################
        # QUESTION 4a END
        ############################################################################

    def forward_kinematics_centre_of_mass(self, joints_readings, up_to_joint=7):
        """This function computes the forward kinematics up to the centre of mass for the given joint frame.
        Reference - Lecture 9 slide 14.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematicks.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint} for the
            centre of mass w.r.t the base of the robot.
        """
        T= np.identity(4)
        T[2, 3] = 0.1575

        T = self.forward_kinematics(joints_readings, up_to_joint-1)
        T = T.dot(self.T_rotationZ(joints_readings[up_to_joint-1]))
        T = T.dot(self.T_translation(self.link_cm[up_to_joint-1, :]))

        return T

    def get_B(self, joint_readings):
        """Given the joint positions of the robot, compute inertia matrix B.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            B (numpy.ndarray): The output is a numpy 7*7 matrix describing the inertia matrix B.
        """
        ############################################################################
        # QUESTION 4b START: implement inside this function
        ############################################################################
        B = np.zeros((7, 7))

        # 对每个link的质心做贡献叠加
        for i in range(1, len(joint_readings) + 1):   # i = 1..7
            jacobian = self.get_jacobian_centre_of_mass(joint_readings, i)

            J_p = jacobian[0:3, :]
            J_o = jacobian[3:6, :]

            m_i = self.mass[i - 1]
            I_link = np.diag(self.Ixyz[i - 1])

            T_com_i = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint=i)
            R_i = T_com_i[0:3, 0:3]
            I_base = R_i @ I_link @ R_i.T

            B += m_i * (J_p.T @ J_p) + (J_o.T @ I_base @ J_o)

        return B
        ############################################################################
        # QUESTION 4b END
        ############################################################################

    def get_C_times_qdot(self, joint_readings, joint_velocities):
        """Given the joint positions and velocities of the robot, compute Coriolis terms C.
        Args:
            joint_readings (list): The positions of the robot joints.
            joint_velocities (list): The velocities of the robot joints.

        Returns:
            C (numpy.ndarray): The output is a numpy 7*1 matrix describing the Coriolis terms C times joint velocities.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        assert isinstance(joint_velocities, list)
        assert len(joint_velocities) == 7
        ############################################################################
        # QUESTION 4c START: implement inside this function
        ############################################################################
        C = np.zeros(7)

        q = np.array(joint_readings)
        qdot = np.array(joint_velocities)

        # 1) 预计算 dB/dq_k（中心差分）
        eps = 1e-6
        dB = []  # dB[k] 是 ∂B/∂q_k, (7,7)

        for k in range(7):
            q_p = q.copy()
            q_m = q.copy()
            q_p[k] += eps
            q_m[k] -= eps

            B_p = self.get_B(q_p.tolist())
            B_m = self.get_B(q_m.tolist())
            dB.append((B_p - B_m) / (2 * eps))

        # 2) 按课件：h_ijk = ∂b_ij/∂q_k - 0.5 * ∂b_jk/∂q_i
        #    c_ij = sum_k h_ijk * qdot_k
        #    (C qdot)_i = sum_j c_ij * qdot_j
        for i in range(7):
            total_i = 0.0
            for j in range(7):
                c_ij = 0.0
                for k in range(7):
                    h_ijk = dB[k][i, j] - 0.5 * dB[i][j, k]
                    c_ij += h_ijk * qdot[k]
                total_i += c_ij * qdot[j]
            C[i] = total_i

        assert isinstance(C, np.ndarray)
        assert C.shape == (7,)
        return C
        ############################################################################
        # QUESTION 4c END
        ############################################################################

    def get_G(self, joint_readings):
        """Given the joint positions of the robot, compute the gravity matrix g.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            G (numpy.ndarray): The output is a numpy 7*1 numpy array describing the gravity matrix g.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        ############################################################################
        # QUESTION 4d START: implement inside this function
        ############################################################################
        g = np.zeros(7)

        # 计算势能 P(q) = sum_i m_i * g * z_i  (z轴朝上时)
        def potential_energy(q_list):
            P = 0.0
            for i in range(1, 8):
                T_com = self.forward_kinematics_centre_of_mass(q_list, up_to_joint=i)
                z_i = T_com[2, 3]
                P += self.mass[i - 1] * self.g * z_i
            return P

        eps = 1e-6
        q = np.array(joint_readings)

        # 中心差分：g_j = dP/dq_j
        for j in range(7):
            q_p = q.copy()
            q_m = q.copy()
            q_p[j] += eps
            q_m[j] -= eps

            P_p = potential_energy(q_p.tolist())
            P_m = potential_energy(q_m.tolist())
            g[j] = (P_p - P_m) / (2.0 * eps)

        assert isinstance(g, np.ndarray)
        assert g.shape == (7,)
        return g
        ############################################################################
        # QUESTION 4d END
        ############################################################################
