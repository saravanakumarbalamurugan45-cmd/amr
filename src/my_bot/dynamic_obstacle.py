#!/usr/bin/env python3
"""
dynamic_obstacle.py

Spawns a small green box in Gazebo and patrols it back and forth in a
straight line, continuously, for as long as this node runs. This makes
the obstacle genuinely DYNAMIC — it's moving the whole time, independent
of when you send a goal — which is what your paper's "dynamic obstacle
scenario" and "Dynamic Clearance Latency" metric actually require.

Run this ONCE at the start of your dynamic-condition batch, in its own
terminal, and leave it running for all dynamic trials. Do NOT restart it
between trials — a continuously patrolling obstacle is what makes the
condition reproducible without hand-timing anything per trial.

────────────────────────────────────────────────────────────────────────
BEFORE RUNNING — set the patrol line to actually cross your goals' paths
────────────────────────────────────────────────────────────────────────
The defaults below are a guess. To find real coordinates: in RViz, hover
your mouse over the open lawn area between where your robot starts and
where you've been sending goals — the bottom-left status bar shows the
(x, y) under the cursor. Pick two points that straddle a path you'd
actually send a goal through.

    -p waypoint_a_x:=... -p waypoint_a_y:=...
    -p waypoint_b_x:=... -p waypoint_b_y:=...

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────
    python3 dynamic_obstacle.py --ros-args \
        -p waypoint_a_x:=0.5  -p waypoint_a_y:=-2.5 \
        -p waypoint_b_x:=0.5  -p waypoint_b_y:=2.5 \
        -p period_s:=8.0
"""

import math
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, SetEntityState, DeleteEntity
from geometry_msgs.msg import Pose


OBSTACLE_NAME = "dynamic_obstacle"

OBSTACLE_SDF = """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <gravity>0</gravity>
      <visual name='visual'>
        <geometry><box><size>0.3 0.3 0.5</size></box></geometry>
        <material>
          <ambient>0 0.8 0.1 1</ambient>
          <diffuse>0 0.8 0.1 1</diffuse>
        </material>
      </visual>
      <collision name='collision'>
        <geometry><box><size>0.3 0.3 0.5</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>"""


class DynamicObstacle(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle')

        self.declare_parameter('waypoint_a_x', 0.5)
        self.declare_parameter('waypoint_a_y', -2.5)
        self.declare_parameter('waypoint_b_x', 0.5)
        self.declare_parameter('waypoint_b_y', 2.5)
        self.declare_parameter('z', 0.25)
        self.declare_parameter('period_s', 8.0)
        self.declare_parameter('update_rate_hz', 10.0)

        self.a = (self.get_parameter('waypoint_a_x').value,
                  self.get_parameter('waypoint_a_y').value)
        self.b = (self.get_parameter('waypoint_b_x').value,
                  self.get_parameter('waypoint_b_y').value)
        self.z = float(self.get_parameter('z').value)
        self.period = float(self.get_parameter('period_s').value)
        rate = float(self.get_parameter('update_rate_hz').value)

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info("Waiting for Gazebo services...")
        self.spawn_cli.wait_for_service()
        self.state_cli.wait_for_service()

        self._spawn_or_reuse()

        self.start_time = time.time()
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"Patrolling between {self.a} and {self.b}, period={self.period}s. "
            f"Leave this running for the whole dynamic-condition batch.")

    def _spawn_or_reuse(self):
        req = SpawnEntity.Request()
        req.name = OBSTACLE_NAME
        req.xml = OBSTACLE_SDF.format(name=OBSTACLE_NAME)
        init_pose = Pose()
        init_pose.position.x, init_pose.position.y, init_pose.position.z = (
            self.a[0], self.a[1], self.z)
        req.initial_pose = init_pose

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            msg = result.status_message if result else "no response"
            if "already" in msg.lower() or "exist" in msg.lower():
                self.get_logger().info(
                    f"'{OBSTACLE_NAME}' already exists in the world — reusing it.")
            else:
                self.get_logger().warn(f"Spawn returned: {msg}")
        else:
            self.get_logger().info(f"Spawned '{OBSTACLE_NAME}'.")

    def tick(self):
        elapsed = time.time() - self.start_time
        cycle = 2.0 * self.period
        phase = elapsed % cycle
        alpha = phase / self.period if phase < self.period else 2.0 - phase / self.period
        x = self.a[0] + alpha * (self.b[0] - self.a[0])
        y = self.a[1] + alpha * (self.b[1] - self.a[1])

        req = SetEntityState.Request()
        req.state.name = OBSTACLE_NAME
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = self.z
        req.state.pose.orientation.w = 1.0
        self.state_cli.call_async(req)  # fire-and-forget, keeps the loop smooth


def main():
    rclpy.init()
    node = DynamicObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
