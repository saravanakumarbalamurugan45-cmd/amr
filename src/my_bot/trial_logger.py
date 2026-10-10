#!/usr/bin/env python3
"""
trial_logger.py

Automatically records one navigation trial's metrics every time a goal is
sent (e.g. RViz "2D Nav Goal" button, or programmatically to /goal_pose).

TRIAL START : a new PoseStamped arrives on /goal_pose
TRIAL END   : Nav2's navigate_to_pose action reaches a terminal status
              (SUCCEEDED / CANCELED / ABORTED)

While a trial is running it buffers, at each ground-truth message:
  - ground-truth pose      (from Gazebo, e.g. /model/<robot>/odometry)
  - estimated pose         (map -> base_link, via tf2, looked up at same time)
  - /cmd_vel linear speed  (used to auto-detect an obstacle-clearance dip)

At trial end it computes and appends ONE row to trials.csv:
  trial_id, condition, success, time_to_goal_s, ate_rmse_m,
  rpe_trans_rmse_mps, rpe_rot_rmse_radps, path_length_m,
  drift_pct, clearance_latency_s, n_gt_samples

────────────────────────────────────────────────────────────────────────
SETUP YOU MUST CHECK BEFORE RUNNING
────────────────────────────────────────────────────────────────────────
1. ground_truth_topic  — must point at your simulator's true pose topic.
   Gazebo Sim (ros_gz_bridge) typically exposes:
       /model/<robot_name>/odometry     (nav_msgs/Odometry)   <- default assumed
   Gazebo Classic often uses:
       /gazebo/model_states              (gazebo_msgs/ModelStates, needs a
                                           small adapter — see note at bottom)
   Check with:  ros2 topic list | grep -i pose
                ros2 topic list | grep -i odom

2. condition — set this per run to whatever you want in the CSV, e.g.
   "static", "dynamic_3obstacles", "dynamic_2obstacles" (after removing one).

3. Run ONE trial at a time (send one goal, wait for it to finish, send the
   next). The logger assumes trials don't overlap.

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────
    chmod +x trial_logger.py
    python3 trial_logger.py --ros-args \
        -p ground_truth_topic:=/model/my_robot/odometry \
        -p condition:=dynamic_2obstacles \
        -p output_csv:=trials.csv

Then just click "2D Nav Goal" in RViz as normal — each trial appends a row
automatically. No need to touch the terminal between trials.
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatusArray
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# action_msgs/msg/GoalStatus status codes
STATUS_EXECUTING = 2
STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6
TERMINAL = {STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED}
STATUS_NAME = {STATUS_SUCCEEDED: "SUCCEEDED", STATUS_CANCELED: "CANCELED",
               STATUS_ABORTED: "ABORTED"}


def quat_angle_diff(q1, q2):
    """Smallest rotation angle (rad) between two (x,y,z,w) quaternions."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(1.0, dot)
    return 2.0 * math.acos(dot)


def umeyama_align(est_xyz: np.ndarray, gt_xyz: np.ndarray):
    """Rigid (rotation + translation, no scale) alignment of est onto gt."""
    est_mean, gt_mean = est_xyz.mean(0), gt_xyz.mean(0)
    est_c, gt_c = est_xyz - est_mean, gt_xyz - gt_mean
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_mean - R @ est_mean
    aligned = (R @ est_xyz.T).T + t
    return aligned


@dataclass
class Sample:
    t: float
    gt_xyz: np.ndarray
    gt_quat: tuple
    est_xyz: np.ndarray
    est_quat: tuple


@dataclass
class TrialState:
    trial_id: int
    start_time: float
    goal_pose: PoseStamped
    tracked_goal_id: object = None
    samples: list = field(default_factory=list)
    speeds: list = field(default_factory=list)  # (t, linear_speed)


class TrialLogger(Node):
    def __init__(self):
        super().__init__('trial_logger')

        self.declare_parameter('ground_truth_topic', '/model/robot/odometry')
        self.declare_parameter('condition', 'unspecified')
        self.declare_parameter('output_csv', 'trials.csv')
        self.declare_parameter('rpe_delta_s', 1.0)
        self.declare_parameter('stop_speed_thresh', 0.05)
        self.declare_parameter('resume_speed_thresh', 0.12)

        self.condition = self.get_parameter('condition').value
        self.output_csv = self.get_parameter('output_csv').value
        self.rpe_delta_s = float(self.get_parameter('rpe_delta_s').value)
        self.stop_thresh = float(self.get_parameter('stop_speed_thresh').value)
        self.resume_thresh = float(self.get_parameter('resume_speed_thresh').value)
        gt_topic = self.get_parameter('ground_truth_topic').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trial: TrialState | None = None
        self.trial_count = 0
        self._ensure_csv_header()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, qos)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status',
                                  self.on_status, qos)
        self.create_subscription(Odometry, gt_topic, self.on_ground_truth, qos)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, qos)

        self.get_logger().info(
            f"trial_logger ready | condition='{self.condition}' | "
            f"ground_truth_topic='{gt_topic}' | writing to '{self.output_csv}'. "
            f"Send a goal (e.g. RViz 2D Nav Goal) to start a trial."
        )

    # ---------------------------------------------------------------- goal
    def on_goal(self, msg: PoseStamped):
        if self.trial is not None:
            self.get_logger().warn(
                "New goal received while a trial was still running — "
                "discarding the previous incomplete trial.")
        self.trial_count += 1
        self.trial = TrialState(
            trial_id=self.trial_count,
            start_time=time.time(),
            goal_pose=msg,
        )
        self.get_logger().info(f"[trial {self.trial_count}] started")

    # -------------------------------------------------------------- status
    def on_status(self, msg: GoalStatusArray):
        if self.trial is None:
            return
        for gs in msg.status_list:
            if self.trial.tracked_goal_id is None and gs.status == STATUS_EXECUTING:
                self.trial.tracked_goal_id = bytes(gs.goal_info.goal_id.uuid)
            if (self.trial.tracked_goal_id is not None
                    and bytes(gs.goal_info.goal_id.uuid) == self.trial.tracked_goal_id
                    and gs.status in TERMINAL):
                self._finish_trial(gs.status)
                return

    # --------------------------------------------------------- ground truth
    def on_ground_truth(self, msg: Odometry):
        if self.trial is None:
            return
        t = self.get_clock().now().nanoseconds / 1e9
        gt_p = msg.pose.pose.position
        gt_q = msg.pose.pose.orientation
        gt_xyz = np.array([gt_p.x, gt_p.y, gt_p.z])
        gt_quat = (gt_q.x, gt_q.y, gt_q.z, gt_q.w)

        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return  # skip this sample rather than crash the trial

        est_p = tf.transform.translation
        est_q = tf.transform.rotation
        est_xyz = np.array([est_p.x, est_p.y, est_p.z])
        est_quat = (est_q.x, est_q.y, est_q.z, est_q.w)

        self.trial.samples.append(Sample(t, gt_xyz, gt_quat, est_xyz, est_quat))

    # -------------------------------------------------------------- cmd_vel
    def on_cmd_vel(self, msg: Twist):
        if self.trial is None:
            return
        speed = math.hypot(msg.linear.x, msg.linear.y)
        self.trial.speeds.append((time.time(), speed))

    # ------------------------------------------------------------- finish
    def _finish_trial(self, status_code):
        tr = self.trial
        self.trial = None  # free the slot immediately
        elapsed = time.time() - tr.start_time
        success = status_code == STATUS_SUCCEEDED

        ate_rmse = rpe_trans = rpe_rot = path_len = drift_pct = float('nan')
        clearance_latency = None
        n = len(tr.samples)

        if n >= 2:
            gt_xyz = np.array([s.gt_xyz for s in tr.samples])
            est_xyz = np.array([s.est_xyz for s in tr.samples])
            times = np.array([s.t for s in tr.samples])

            aligned_est = umeyama_align(est_xyz, gt_xyz)
            errs = np.linalg.norm(aligned_est - gt_xyz, axis=1)
            ate_rmse = float(np.sqrt(np.mean(errs ** 2)))

            path_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
            drift_pct = (ate_rmse / path_len * 100.0) if path_len > 1e-6 else float('nan')

            # RPE at a fixed delta (matches m/s, rad/s units)
            trans_errs, rot_errs = [], []
            for i in range(n):
                target_t = times[i] + self.rpe_delta_s
                j = int(np.argmin(np.abs(times - target_t)))
                if j <= i or abs(times[j] - target_t) > 0.3 * self.rpe_delta_s:
                    continue
                gt_rel = gt_xyz[j] - gt_xyz[i]
                est_rel = aligned_est[j] - aligned_est[i]
                trans_errs.append(np.linalg.norm(gt_rel - est_rel))
                rot_errs.append(quat_angle_diff(tr.samples[i].gt_quat,
                                                 tr.samples[i].est_quat))
            if trans_errs:
                rpe_trans = float(np.sqrt(np.mean(np.square(trans_errs))))
                rpe_rot = float(np.sqrt(np.mean(np.square(rot_errs))))

        # --- obstacle-clearance latency: detect a genuine stop-after-moving
        # event, NOT the robot's initial ramp-up from standing start at t=0.
        # Requires the robot to have already been moving normally for
        # REQUIRED_MOVING_S before a subsequent dip counts as a real stop,
        # and requires the resumed speed to hold for REQUIRED_RESUME_S
        # before counting as "cleared" (filters out brief noise/jitter).
        REQUIRED_MOVING_S = 1.5
        REQUIRED_RESUME_S = 0.5

        clearance_latency = None
        established = False
        established_since = None
        stop_t = None
        resumed_since = None

        for t, v in tr.speeds:
            if not established:
                if v > self.resume_thresh:
                    if established_since is None:
                        established_since = t
                    elif t - established_since >= REQUIRED_MOVING_S:
                        established = True
                else:
                    established_since = None
                continue

            if stop_t is None:
                if v < self.stop_thresh:
                    stop_t = t
                    resumed_since = None
            else:
                if v > self.resume_thresh:
                    if resumed_since is None:
                        resumed_since = t
                    elif t - resumed_since >= REQUIRED_RESUME_S:
                        clearance_latency = resumed_since - stop_t
                        break
                else:
                    resumed_since = None

        self._write_row(tr, success, status_code, elapsed, ate_rmse, rpe_trans,
                         rpe_rot, path_len, drift_pct, clearance_latency, n)

        self.get_logger().info(
            f"[trial {tr.trial_id}] finished: {STATUS_NAME.get(status_code, status_code)} | "
            f"t={elapsed:.1f}s | ATE={ate_rmse:.3f}m | samples={n} -> logged")

    # ---------------------------------------------------------------- csv
    def _ensure_csv_header(self):
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'trial_id', 'condition', 'success', 'status',
                    'time_to_goal_s', 'ate_rmse_m', 'rpe_trans_rmse_mps',
                    'rpe_rot_rmse_radps', 'path_length_m', 'drift_pct',
                    'clearance_latency_s', 'n_gt_samples',
                    'goal_x', 'goal_y',
                ])

    def _write_row(self, tr, success, status_code, elapsed, ate, rpe_t, rpe_r,
                    path_len, drift, clearance, n):
        with open(self.output_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                tr.trial_id, self.condition, success,
                STATUS_NAME.get(status_code, status_code),
                round(elapsed, 3), round(ate, 4), round(rpe_t, 4),
                round(rpe_r, 4), round(path_len, 3), round(drift, 3),
                round(clearance, 3) if clearance is not None else 'NA',
                n, tr.goal_pose.pose.position.x, tr.goal_pose.pose.position.y,
            ])


def main():
    rclpy.init()
    node = TrialLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# ────────────────────────────────────────────────────────────────────────
# NOTE — if your simulator only publishes gazebo_msgs/ModelStates instead
# of a per-robot Odometry topic, add a tiny bridge node beforehand that
# republishes the robot's entry from ModelStates as nav_msgs/Odometry on
# the topic you pass as ground_truth_topic. Ask if you want that written
# too — it's about 25 lines.
# ────────────────────────────────────────────────────────────────────────
