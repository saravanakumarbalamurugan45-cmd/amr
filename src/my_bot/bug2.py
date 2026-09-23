#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class Bug2Planner(Node):
    def __init__(self):
        super().__init__('bug2_planner')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # Topic Subscriptions & Publishers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.start_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_trajectory_pub = self.create_publisher(Path, '/bug2_actual_trajectory', 10)

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Control Loop Timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        # Internal State Parameters
        self.start_pose = None
        self.goal_pose = None
        self.regions = None
        self.state = 'GO_TO_GOAL'  # States: 'GO_TO_GOAL', 'WALL_FOLLOW'

        self.obstacle_dist_threshold = 0.5
        self.wall_dist_target = 0.45
        self.goal_reached_threshold = 0.35

        # Bug 2 M-Line Parameters
        self.hit_point = None
        self.hit_dist_to_goal = float('inf')
        self.m_line_tolerance = 0.12  # meters from line

        # Trajectory Metrics
        self.actual_trajectory_msg = Path()
        self.actual_trajectory_msg.header.frame_id = 'map'
        self.prev_map_pose = None
        self.actual_distance_traveled = 0.0
        self.planning_start_time = None
        self.is_tracking_goal = False

    def get_current_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def start_callback(self, msg):
        self.get_logger().info("Start pose updated.")

    def scan_callback(self, msg):
        def get_min_range(start_deg, end_deg):
            n = len(msg.ranges)
            i_start = int((start_deg % 360) / 360.0 * n)
            i_end = int((end_deg % 360) / 360.0 * n)
            sub = msg.ranges[i_start:i_end] if i_start < i_end else msg.ranges[i_start:] + msg.ranges[:i_end]
            valid = [r for r in sub if not math.isnan(r) and not math.isinf(r) and r > msg.range_min]
            return min(valid) if valid else msg.range_max

        self.regions = {
            'front':  get_min_range(330, 30),
            'fleft':  get_min_range(30, 75),
            'left':   get_min_range(75, 105),
            'fright': get_min_range(285, 330),
            'right':  get_min_range(255, 285),
        }

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def goal_callback(self, msg):
        self.goal_pose = msg
        pose = self.get_robot_pose()
        if pose is not None:
            self.start_pose = (pose[0], pose[1])
        else:
            self.start_pose = (0.0, 0.0)

        self.actual_trajectory_msg = Path()
        self.actual_trajectory_msg.header.frame_id = 'map'
        self.actual_distance_traveled = 0.0
        self.planning_start_time = self.get_current_sec()
        self.is_tracking_goal = True
        self.state = 'GO_TO_GOAL'
        self.get_logger().info(f"Bug2 Goal Received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def distance_to_m_line(self, rx, ry):
        if self.start_pose is None or self.goal_pose is None:
            return float('inf')
        x1, y1 = self.start_pose
        x2 = self.goal_pose.pose.position.x
        y2 = self.goal_pose.pose.position.y

        numerator = abs((y2 - y1) * rx - (x2 - x1) * ry + x2 * y1 - y2 * x1)
        denominator = math.hypot(y2 - y1, x2 - x1)
        return numerator / denominator if denominator != 0 else 0.0

    def control_loop(self):
        if not self.is_tracking_goal or self.goal_pose is None or self.regions is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, ryaw = pose

        if self.prev_map_pose is not None:
            step_dist = math.hypot(rx - self.prev_map_pose[0], ry - self.prev_map_pose[1])
            if step_dist > 0.01:
                self.actual_distance_traveled += step_dist
                self.prev_map_pose = (rx, ry)
        else:
            self.prev_map_pose = (rx, ry)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = rx
        pose_stamped.pose.position.y = ry
        self.actual_trajectory_msg.poses.append(pose_stamped)
        self.actual_trajectory_pub.publish(self.actual_trajectory_msg)

        gx = self.goal_pose.pose.position.x
        gy = self.goal_pose.pose.position.y
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        goal_angle = math.atan2(gy - ry, gx - rx)
        heading_error = math.atan2(math.sin(goal_angle - ryaw), math.cos(goal_angle - ryaw))

        if dist_to_goal <= self.goal_reached_threshold:
            self.stop_robot()
            self.is_tracking_goal = False
            time_to_reach = self.get_current_sec() - self.planning_start_time
            print(f"\n[BUG2] Goal Reached! Distance: {self.actual_distance_traveled:.2f}m, Time: {time_to_reach:.2f}s")
            return

        twist = Twist()

        if self.state == 'GO_TO_GOAL':
            if self.regions['front'] < self.obstacle_dist_threshold:
                self.state = 'WALL_FOLLOW'
                self.hit_point = (rx, ry)
                self.hit_dist_to_goal = dist_to_goal
            else:
                if abs(heading_error) > 0.2:
                    twist.angular.z = 0.5 if heading_error > 0 else -0.5
                else:
                    twist.linear.x = 0.22
                    twist.angular.z = 0.8 * heading_error

        elif self.state == 'WALL_FOLLOW':
            dist_to_hit = math.hypot(rx - self.hit_point[0], ry - self.hit_point[1])
            dist_m_line = self.distance_to_m_line(rx, ry)

            # Departure condition: On M-line, closer to goal than hit point, and path to goal is open
            if dist_to_hit > 0.5 and dist_m_line < self.m_line_tolerance and dist_to_goal < self.hit_dist_to_goal:
                if self.regions['front'] > (self.obstacle_dist_threshold + 0.2):
                    self.state = 'GO_TO_GOAL'

            if self.state == 'WALL_FOLLOW':
                if self.regions['front'] < self.obstacle_dist_threshold:
                    twist.angular.z = 0.5
                elif self.regions['fright'] < self.wall_dist_target:
                    twist.linear.x = 0.1
                    twist.angular.z = 0.3
                elif self.regions['right'] > (self.wall_dist_target + 0.15):
                    twist.linear.x = 0.12
                    twist.angular.z = -0.3
                else:
                    twist.linear.x = 0.18

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = Bug2Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
