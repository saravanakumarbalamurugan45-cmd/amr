import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Precise topic remappings for the core SLAM node
    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('depth/image', '/camera/depth/image_raw'),
        ('rgb/camera_info', '/camera/depth/camera_info'),
        ('grid_map', '/map'),
    ]

    parameters = [
        {
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'publish_tf': True,
            'use_sim_time': use_sim_time,
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': False,
            'subscribe_scan_cloud': False,
            # --- FIXED: RAW INTEGERS TO PASS ROS2 PARAMETER TYPE VALIDATION ---
            'qos_image': 1,  # 1 = SensorData/BestEffort profile match
            'qos_depth': 1,  # 1 = SensorData/BestEffort profile match
            'qos_camera_info': 1,  # 1 = SensorData/BestEffort profile match
            'qos_odom': 1,  # 1 = SensorData/BestEffort profile match
            'approx_sync': True,
            'approx_sync_max_interval': 0.05,  # Tightened interval to sync simulation loops
            'sync_queue_size': 100,  # Expanded queue to hold delayed sensor packets
            'Reg/Strategy': '0',
            'Vis/MinInliers': '10',
            'Vis/FeatureType': '8',
            'Grid/3D': 'true',
            'Grid/FromDepth': 'true',
            'Grid/CellSize': '0.05',
            'Grid/RayTracing': 'true',
            'Grid/3DObstacleClearance': 'true',
            'Grid/NormalsSegmentation': 'false',
            'Grid/MinGroundHeight': '-0.5',
            'Grid/MaxObstacleHeight': '2.5',
            'Grid/RangeMin': '0.1',
            'Grid/RangeMax': '5.0',
            'Cloud/VoxelSize': '0.05',
            'Cloud/MaxDepth': '5.0',
            'Cloud/MinDepth': '0.1',
        }
    ]

    # 1. Main Core Visual SLAM Node
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        arguments=['--delete_db_on_start'],
    )

    # 2. PointCloud Generator utilizing official ROS2 image_rect parameter mappings
    rtabmap_cloud_node = Node(
        package='rtabmap_util',
        executable='point_cloud_xyzrgb',
        name='point_cloud_xyzrgb',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'approx_sync': True}],
        # Maps your custom raw camera fields directly to rtabmap's expected tracking keys
        remappings=[
            ('rgb/image_rect', '/camera/image_raw'),
            ('depth/image_rect', '/camera/depth/image_raw'),
            ('rgb/camera_info', '/camera/depth/camera_info'),
            ('/cloud_map', '/rtabmap/cloud_map'),
        ],
    )

    # 3. Native Visualizer GUI Node
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=parameters,
        remappings=remappings,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock',
        ),
        rtabmap_node,
        rtabmap_cloud_node,
        rtabmap_viz_node,
    ])
