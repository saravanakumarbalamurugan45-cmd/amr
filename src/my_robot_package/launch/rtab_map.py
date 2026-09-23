import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # 1. OpenCV Depth Filtering Node
    depth_filter_node = Node(
        package='my_robot_package',  # Ensure this matches package name in setup.py
        executable='depth_filter_node',
        name='depth_filter_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    remappings = [
        ('rgb/image', '/camera/image_raw'),          
        ('depth/image', '/camera/depth/image_filtered'),
        ('rgb/camera_info', '/camera/depth/camera_info'),   
        ('grid_map', '/map')                          
    ]

    parameters = [{
        'frame_id': 'base_link',
        'odom_frame_id': 'odom',
        'map_frame_id': 'map',
        'publish_tf': True,                    
        'use_sim_time': use_sim_time,
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'subscribe_scan': False,              
        'subscribe_scan_cloud': False,        
        'qos_image': 1,                       
        'qos_depth': 1,                       
        'qos_camera_info': 1,                 
        'qos_odom': 1,                        
        'approx_sync': True,
        'approx_sync_max_interval': 0.05,     
        'sync_queue_size': 100,               
        'Reg/Strategy': '0',                  
        'Vis/MinInliers': '10',               
        'Vis/FeatureType': '8',               
        'Grid/3D': 'true',                    
        'Grid/FromDepth': 'true',             
        'Grid/CellSize': '0.05',              
        'Grid/RayTracing': 'true',            
        'Grid/3DObstacleClearance': 'true',   
        'Grid/NormalsSegmentation': 'false', 
        'Grid/MinGroundHeight': '0.08',       
        'Grid/MaxObstacleHeight': '2.5',
        'Grid/RangeMin': '0.1',
        'Grid/RangeMax': '5.0',
        'Cloud/VoxelSize': '0.05',             
        'Cloud/MaxDepth': '5.0',
        'Cloud/MinDepth': '0.1'
    }]

    # 2. Main RTAB-Map SLAM Node
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        arguments=['--delete_db_on_start']     
    )

    # 3. PointCloud Generator reading /camera/depth/image_filtered
    rtabmap_cloud_node = Node(
        package='rtabmap_util',
        executable='point_cloud_xyzrgb',
        name='point_cloud_xyzrgb',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'approx_sync': True}],
        remappings=[
            ('rgb/image_rect', '/camera/image_raw'),          
            ('depth/image_rect', '/camera/depth/image_filtered'),
            ('rgb/camera_info', '/camera/depth/camera_info'),
            ('/cloud_map', '/rtabmap/cloud_map')
        ]
    )

    # 4. Native Visualizer Node
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=parameters,
        remappings=remappings
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),
        depth_filter_node,
        rtabmap_node,
        rtabmap_cloud_node,                     
        rtabmap_viz_node
    ])
