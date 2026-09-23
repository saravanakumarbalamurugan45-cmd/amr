#!/usr/bin/env python3

import math
import time
import heapq

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav2_msgs.action import NavigateThroughPoses
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')
        
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
            
        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.start_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/dijkstra_path', 10)
        self.actual_trajectory_pub = self.create_publisher(Path, '/dijkstra_actual_trajectory', 10)
        self.nav2_plan_pub = self.create_publisher(Path, '/plan', 10)
        self.expansion_pub = self.create_publisher(MarkerArray, '/dijkstra_expansion', 10)

        # Action Client for Nav2 Navigation
        self.nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        
        # TF2 Listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Movement tracking timer (10 Hz)
        self.timer = self.create_timer(0.1, self.track_robot_position)
        
        # Internal state
        self.map_data = None
        self.start_pose = None
        self.goal_pose = None
        self.map_received = False
        
        # Trajectory & Navigation metrics tracking
        self.actual_trajectory_msg = Path()
        self.actual_trajectory_msg.header.frame_id = 'map'
        self.prev_map_pose = None
        self.actual_distance_traveled = 0.0
        self.planning_start_time = None
        self.goal_reached_threshold = 0.35
        self.is_tracking_goal = False

    def get_current_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def map_callback(self, msg):
        if not self.map_received:
            self.map_received = True
            now_sec = self.get_current_sec()
            msg_stamp_sec = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            map_gen_time = max(0.01, now_sec - msg_stamp_sec)
            
            print("\n" + "="*55)
            print("                1. MAP METRICS")
            print("="*55)
            print(f" Time to Generate/Receive Map : {map_gen_time:.2f} seconds")
            print("="*55 + "\n")
        
        self.map_data = msg

    def start_callback(self, msg):
        self.start_pose = msg.pose.pose
        self.get_logger().info("Start Pose updated from RViz.")

    def get_robot_map_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return transform.transform.translation
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def track_robot_position(self):
        current_pos = self.get_robot_map_pose()
        if current_pos is None:
            return

        if self.is_tracking_goal:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = current_pos.x
            pose_stamped.pose.position.y = current_pos.y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            self.actual_trajectory_msg.poses.append(pose_stamped)
            self.actual_trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            self.actual_trajectory_pub.publish(self.actual_trajectory_msg)

            if self.prev_map_pose is not None:
                dx = current_pos.x - self.prev_map_pose.x
                dy = current_pos.y - self.prev_map_pose.y
                dist = math.hypot(dx, dy)
                if dist > 0.01:
                    self.actual_distance_traveled += dist
                    self.prev_map_pose = current_pos
            else:
                self.prev_map_pose = current_pos

            dist_to_goal = math.hypot(
                self.goal_pose.pose.position.x - current_pos.x,
                self.goal_pose.pose.position.y - current_pos.y
            )

            time_elapsed = self.get_current_sec() - self.planning_start_time
            print(f"\r[NAVIGATING] Dist to Goal: {dist_to_goal:.2f} m | Distance Covered: {self.actual_distance_traveled:.2f} m | Time: {time_elapsed:.1f} s", end="")

            if dist_to_goal <= self.goal_reached_threshold and time_elapsed > 1.0:
                time_to_reach = self.get_current_sec() - self.planning_start_time
                self.is_tracking_goal = False

                print("\n\n" + "="*55)
                print("           3. ROBOT NAVIGATION METRICS")
                print("="*55)
                print(f" Distance Covered by Robot : {self.actual_distance_traveled:.2f} meters")
                print(f" Time to Reach Goal        : {time_to_reach:.2f} seconds")
                print("="*55 + "\n")

    def goal_callback(self, msg):
        self.goal_pose = msg
        if self.map_data is None:
            self.get_logger().warn("Map data missing!")
            return
            
        curr_pos = self.get_robot_map_pose()
        if curr_pos is not None:
            from geometry_msgs.msg import Pose
            self.start_pose = Pose()
            self.start_pose.position.x = curr_pos.x
            self.start_pose.position.y = curr_pos.y
        elif self.start_pose is None:
            self.get_logger().warn("Start pose missing! Set '2D Pose Estimate' in RViz.")
            return

        self.actual_trajectory_msg = Path()
        self.actual_trajectory_msg.header.frame_id = 'map'
        self.actual_distance_traveled = 0.0
        self.prev_map_pose = curr_pos
        self.planning_start_time = self.get_current_sec()
        self.is_tracking_goal = True
        
        self.plan_path()

    def world_to_map(self, wx, wy):
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        res = self.map_data.info.resolution
        return int((wx - origin_x) / res), int((wy - origin_y) / res)

    def map_to_world(self, mx, my):
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        res = self.map_data.info.resolution
        return origin_x + (mx + 0.5) * res, origin_y + (my + 0.5) * res

    def calculate_path_length(self, path_msg):
        total_length = 0.0
        poses = path_msg.poses
        for i in range(1, len(poses)):
            p1 = poses[i-1].pose.position
            p2 = poses[i].pose.position
            total_length += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return total_length

    def publish_expansion_visualization(self, visited_nodes):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'dijkstra_search'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.map_data.info.resolution
        marker.scale.y = self.map_data.info.resolution
        marker.scale.z = 0.02
        
        # Transparent Purple/Magenta
        marker.color.r = 0.7
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.35

        for node in visited_nodes:
            wx, wy = self.map_to_world(node[0], node[1])
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.01
            marker.points.append(p)

        marker_array.markers.append(marker)
        self.expansion_pub.publish(marker_array)

    def execute_nav2_goal(self, path_msg):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 Action Server 'navigate_through_poses' is unavailable!")
            return

        goal_msg = NavigateThroughPoses.Goal()
        poses = path_msg.poses
        for i in range(len(poses)):
            p = poses[i]
            if i < len(poses) - 1:
                next_p = poses[i+1]
                yaw = math.atan2(next_p.pose.position.y - p.pose.position.y, 
                                 next_p.pose.position.x - p.pose.position.x)
            else:
                yaw = 0.0
            p.pose.orientation.z = math.sin(yaw / 2.0)
            p.pose.orientation.w = math.cos(yaw / 2.0)
            goal_msg.poses.append(p)

        self.nav_client.send_goal_async(goal_msg)

    def plan_path(self):
        t_start_plan = time.perf_counter()
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        grid = self.map_data.data
        
        start_mx, start_my = self.world_to_map(self.start_pose.position.x, self.start_pose.position.y)
        goal_mx, goal_my = self.world_to_map(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
        
        start = (start_mx, start_my)
        goal = (goal_mx, goal_my)
        
        # Priority Queue driven purely by g_score (actual path cost)
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        
        came_from = {start: None}
        g_score = {start: 0.0}
        
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]
        
        found = False
        while open_set:
            current_g, current = heapq.heappop(open_set)
            
            if current == goal:
                found = True
                break
                
            if current_g > g_score[current]:
                continue

            for dx, dy, cost in moves:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    idx = ny * width + nx
                    if grid[idx] == 0 or grid[idx] == -1:
                        neighbor = (nx, ny)
                        tentative_g = current_g + cost
                        
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            heapq.heappush(open_set, (tentative_g, neighbor))
        
        comp_time = (time.perf_counter() - t_start_plan) * 1000.0
        self.publish_expansion_visualization(came_from.keys())

        if not found:
            self.get_logger().error("Dijkstra Search Failed: Path blocked or goal in obstacle!")
            self.is_tracking_goal = False
            return
            
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        curr = goal
        while curr is not None:
            wx, wy = self.map_to_world(curr[0], curr[1])
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            
            path_msg.poses.append(pose)
            curr = came_from[curr]
            
        path_msg.poses.reverse()
        path_length = self.calculate_path_length(path_msg)
        
        self.path_pub.publish(path_msg)
        self.nav2_plan_pub.publish(path_msg)
        
        print("\n" + "="*55)
        print("        2. DIJKSTRA PATH PLANNING METRICS")
        print("="*55)
        print(f" Total Path Length     : {path_length:.2f} meters")
        print(f" Computation Time      : {comp_time:.2f} ms")
        print(f" Total Path Waypoints  : {len(path_msg.poses)}")
        print(f" Explored Nodes Count  : {len(came_from)} cells")
        print("="*55 + "\n")

        self.execute_nav2_goal(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DijkstraPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
