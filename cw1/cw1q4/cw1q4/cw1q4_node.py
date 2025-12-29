#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from std_msgs.msg import Float64

# ###################### STUDENT CODE START (IMPORT SERVICE) ######################
from cw1q4_interfaces.srv import QuatToEuler, QuatToRodrigues
import math

# ####################### STUDENT CODE END (IMPORT SERVICE) #######################

class QuatToEulerService(Node):
    def __init__(self):
        super().__init__('quat_to_euler_service_node', start_parameter_services=False)
        self.srv = self.create_service(QuatToEuler, 'quat_to_euler', self.quat_to_euler_callback)

    def quat_to_euler_callback(self, request, response):
        q_x, q_y, q_z, q_w = request.q.x, request.q.y, request.q.z, request.q.w
        self.get_logger().info(f'Euler service received quaternion: [w={q_w}, x={q_x}, y={q_y}, z={q_z}]')
        response.z, response.y, response.x = Float64(), Float64(), Float64()

        # ###################### STUDENT CODE START (QUATERNION TO EULER) ######################
        response.z.data = math.atan2(2 * (q_w*q_z + q_x*q_y), 1 - 2 * (q_y*q_y + q_z*q_z))
        response.y.data = math.asin(2 * (q_w*q_y - q_z*q_x))
        response.x.data = math.atan2(2 * (q_w*q_x + q_y*q_z), 1 - 2 * (q_x*q_x + q_y*q_y))

        # ####################### STUDENT CODE END (QUATERNION TO EULER) #######################

        self.get_logger().info(f'Calculated Euler Angles: [z={response.z.data}, y={response.y.data}, x={response.x.data}]')
        return response

class QuatToRodriguesService(Node):
    def __init__(self):
        super().__init__('quat_to_rodrigues_service_node', start_parameter_services=False)
        self.srv = self.create_service(QuatToRodrigues, 'quat_to_rodrigues', self.quat_to_rodrigues_callback)

    def quat_to_rodrigues_callback(self, request, response):
        q_x, q_y, q_z, q_w = request.q.x, request.q.y, request.q.z, request.q.w
        self.get_logger().info(f'Rodrigues service received quaternion: [w={q_w}, x={q_x}, y={q_y}, z={q_z}]')
        response.x, response.y, response.z = Float64(), Float64(), Float64()

        # ###################### STUDENT CODE START (QUATERNION TO RODRIGUES) ##################
        theta = 2 * math.acos(q_w)
        den = math.sqrt(q_x*q_x + q_y*q_y + q_z*q_z)

        if den < 1e-6:   #avoid 0
            response.x.data = 0.0
            response.y.data = 0.0
            response.z.data = 0.0
        else:
            response.x.data = q_x / den * theta
            response.y.data = q_y / den * theta
            response.z.data = q_z / den * theta

        # ####################### STUDENT CODE END (QUATERNION TO RODRIGUES) ###################

        self.get_logger().info(f'Calculated Rodrigues Vector: [x={response.x.data}, y={response.y.data}, z={response.z.data}]')
        return response

def main(args=None):
    rclpy.init(args=args)
    try:
        # Create instances of both service nodes
        quat_to_euler_node = QuatToEulerService()
        quat_to_rodrigues_node = QuatToRodriguesService()

        # Use a MultiThreadedExecutor to handle callbacks from both services concurrently
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(quat_to_euler_node)
        executor.add_node(quat_to_rodrigues_node)

        print("Both Quaternion conversion services are ready.")

        try:
            executor.spin()
        finally:
            executor.shutdown()
            quat_to_euler_node.destroy_node()
            quat_to_rodrigues_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()