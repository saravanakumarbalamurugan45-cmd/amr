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

        # Subscribers for synchronized or individual image processing
        self.rgb_sub = self.create_subscription(
            Image, '/camera/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publisher for filtered depth image
        self.filtered_depth_pub = self.create_publisher(
            Image, '/camera/depth/image_filtered', 10)

        self.latest_rgb = None

        # HSV Color Range for Grass Detection (Adjustable for your Gazebo texture)
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        if self.latest_rgb is None:
            # Pass through raw depth until RGB is ready
            self.filtered_depth_pub.publish(msg)
            return

        try:
            # Convert ROS Depth image to OpenCV numpy array (32-bit float or 16-bit int)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_copy = depth_image.copy()

            # 1. Convert RGB frame to HSV for color-based feature mask
            hsv = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # 2. Find small contours (pebbles, small ground debris) using edge detection
            gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            small_object_mask = np.zeros_like(gray)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # If contour area is small (e.g. smaller than 150 pixels), mark as non-obstacle
                if 0 < area < 150:
                    cv2.drawContours(small_object_mask, [cnt], -1, 255, -1)

            # Combine grass mask and small object mask into a traversable mask
            traversable_mask = cv2.bitwise_or(grass_mask, small_object_mask)

            # 3. Clear depth values (set to NaN or 0) for grass / small traversable objects
            if depth_copy.dtype == np.float32:
                depth_copy[traversable_mask > 0] = np.nan
            else: # 16UC1
                depth_copy[traversable_mask > 0] = 0

            # Publish updated depth image
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
