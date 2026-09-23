#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
import math
import time

class DFSNavNode(Node):
    def __init__(self):
        super().__init__('dfs_nav_node')
        
        self.start_time_init = time.time()

        # 1. Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # 2. TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Action Client for Nav2
        self.nav_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        
        # State Variables
        self.map_data = None
        self.map_info = None
        self.map_received = False
        
        # Navigation Tracking
        self.nav_start_time = None
        self.goal_world_pos = None
        self.last_robot_pos = None
        self.total_distance_covered = 0.0
        self.tracking_timer = None
        self.is_navigating = False

        self.get_logger().info("DFS Navigation Node initialized. Waiting for /map and /goal_pose...")

    def map_callback(self, msg):
        if not self.map_received:
            map_time = time.time() - self.start_time_init
            print(f"Time to Generate/Receive Map : {map_time:.2f} seconds")
            print("=========================================================================")
            self.map_received = True

        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def world_to_grid(self, wx, wy):
        col = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col

    def grid_to_world(self, row, col):
        wx = self.map_info.origin.position.x + (col + 0.5) * self.map_info.resolution
        wy = self.map_info.origin.position.y + (row + 0.5) * self.map_info.resolution
        return wx, wy

    def goal_callback(self, msg):
        if not self.map_received:
            self.get_logger().warn("Goal received, but /map is not available yet!")
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().error("Unable to determine robot start position from TF!")
            return

        self.get_logger().info("Start Pose updated from RViz.")
        self.execute_navigation(robot_pose, (msg.pose.position.x, msg.pose.position.y))

    def run_dfs(self, start_grid, goal_grid):
        rows, cols = self.map_data.shape
        
        def is_safe(r, c):
            if not (0 <= r < rows and 0 <= c < cols):
                return False
            val = self.map_data[r, c]
            return val < 50 and val != 100

        stack = [start_grid]
        visited = set([start_grid])
        parent = {}
        nodes_explored = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        found = False
        while stack:
            curr = stack.pop()
            nodes_explored += 1
            
            if curr == goal_grid:
                found = True
                break

            r, c = curr
            sorted_dirs = sorted(
                directions, 
                key=lambda d: math.hypot((r + d[0]) - goal_grid[0], (c + d[1]) - goal_grid[1]), 
                reverse=True
            )

            for dr, dc in sorted_dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    if is_safe(nr, nc) or (nr, nc) == goal_grid:
                        visited.add((nr, nc))
                        parent[(nr, nc)] = curr
                        stack.append((nr, nc))

        if not found:
            return [], nodes_explored

        path = []
        curr = goal_grid
        while curr != start_grid:
            path.append(curr)
            curr = parent[curr]
        path.append(start_grid)
        path.reverse()
        return path, nodes_explored

    def interpolate_waypoints(self, raw_world_pts, step_dist=0.25):
        interpolated = []
        for i in range(len(raw_world_pts) - 1):
            p1 = np.array(raw_world_pts[i])
            p2 = np.array(raw_world_pts[i + 1])
            dist = np.linalg.norm(p2 - p1)
            num_steps = max(int(dist / step_dist), 1)
            for t in np.linspace(0, 1, num_steps, endpoint=False):
                pt = p1 + t * (p2 - p1)
                interpolated.append((pt[0], pt[1]))
        interpolated.append(raw_world_pts[-1])
        return interpolated

    def execute_navigation(self, start_world, goal_world):
        self.goal_world_pos = goal_world
        self.last_robot_pos = start_world
        self.total_distance_covered = 0.0

        start_grid = self.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.world_to_grid(goal_world[0], goal_world[1])

        t_start = time.time()
        grid_path, explored_count = self.run_dfs(start_grid, goal_grid)
        comp_time_ms = (time.time() - t_start) * 1000.0

        if not grid_path:
            self.get_logger().error("DFS failed to find a path!")
            return

        raw_world_pts = [self.grid_to_world(r, c) for r, c in grid_path]
        dense_pts = self.interpolate_waypoints(raw_world_pts, step_dist=0.25)

        path_length = sum(
            math.hypot(raw_world_pts[i+1][0] - raw_world_pts[i][0], raw_world_pts[i+1][1] - raw_world_pts[i][1])
            for i in range(len(raw_world_pts) - 1)
        )

        print("=========================================================================")
        print("         2. DFS PATH PLANNING METRICS")
        print("=========================================================================")
        print(f"Total Path Length     : {path_length:.2f} meters")
        print(f"Computation Time      : {comp_time_ms:.2f} ms")
        print(f"Total Path Waypoints  : {len(dense_pts)}")
        print(f"Explored Nodes Count  : {explored_count} cells")
        print("=========================================================================\n")

        goal_msg = NavigateThroughPoses.Goal()
        now = self.get_clock().now().to_msg()

        for i in range(len(dense_pts)):
            wx, wy = dense_pts[i]
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = now
            pose.pose.position.x = wx
            pose.pose.position.y = wy

            if i < len(dense_pts) - 1:
                yaw = math.atan2(dense_pts[i+1][1] - wy, dense_pts[i+1][0] - wx)
            else:
                yaw = 0.0

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            goal_msg.poses.append(pose)

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 Action Server unavailable!")
            return

        self.nav_start_time = time.time()
        self.is_navigating = True
        
        self.nav_client.send_goal_async(goal_msg)

        # Start tracking thread loop directly
        if self.tracking_timer:
            self.tracking_timer.cancel()
        self.tracking_timer = self.create_timer(0.2, self.track_navigation_progress)

    def track_navigation_progress(self):
        if not self.is_navigating:
            return

        curr_pose = self.get_robot_pose()
        if curr_pose is None or self.last_robot_pos is None:
            return

        step_dist = math.hypot(curr_pose[0] - self.last_robot_pos[0], curr_pose[1] - self.last_robot_pos[1])
        if step_dist > 0.01:
            self.total_distance_covered += step_dist
            self.last_robot_pos = curr_pose

        dist_to_goal = math.hypot(self.goal_world_pos[0] - curr_pose[0], self.goal_world_pos[1] - curr_pose[1])
        elapsed_time = time.time() - self.nav_start_time

        print(f"[NAVIGATING] Dist to Goal: {dist_to_goal:.2f} m | Distance Covered: {self.total_distance_covered:.2f} m | Time: {elapsed_time:.1f} s", end='\r')

        # Threshold check: when robot reaches within 0.35m of goal
        if dist_to_goal <= 0.35 and elapsed_time > 1.0:
            self.is_navigating = False
            self.tracking_timer.cancel()
            total_time = time.time() - self.nav_start_time

            print("\n=========================================================================")
            print("         3. ROBOT NAVIGATION METRICS")
            print("=========================================================================")
            print(f"Distance Covered by Robot : {self.total_distance_covered:.2f} meters")
            print(f"Time to Reach Goal        : {total_time:.2f} seconds")
            print("=========================================================================\n")

def main(args=None):
    rclpy.init(args=args)
    node = DFSNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
