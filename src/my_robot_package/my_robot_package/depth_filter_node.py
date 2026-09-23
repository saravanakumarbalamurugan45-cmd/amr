#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthFilterNode(Node):
    def __init__(self):
        super().__init__('depth_filter_node')
        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publisher for cleaned depth image
        self.filtered_depth_pub = self.create_publisher(
            Image, '/camera/depth/image_filtered', 10)

        self.latest_rgb = None

        # HSV Range for Grass Detection (Tune upper/lower values as needed)
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        if self.latest_rgb is None:
            self.filtered_depth_pub.publish(msg)
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_copy = depth_image.copy()

            # 1. Color Masking (Grass)
            hsv = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # 2. Small Object Contour Masking
            gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            small_object_mask = np.zeros_like(gray)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 0 < area < 150: # Filters out tiny ground artifacts
                    cv2.drawContours(small_object_mask, [cnt], -1, 255, -1)

            # Combine Masks
            traversable_mask = cv2.bitwise_or(grass_mask, small_object_mask)

            # 3. Set Traversable Pixels to NaN/0 (Ignored by point cloud generator)
            if depth_copy.dtype == np.float32:
                depth_copy[traversable_mask > 0] = np.nan
            else:
                depth_copy[traversable_mask > 0] = 0

            filtered_msg = self.bridge.cv2_to_imgmsg(depth_copy, encoding=msg.encoding)
            filtered_msg.header = msg.header
            self.filtered_depth_pub.publish(filtered_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
