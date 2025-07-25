from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Include other files if needed (world files, config, etc.)
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aldon',
    maintainer_email='aldon@gmail.com',
    description='Robot ROS 2 node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_node = my_robot.robot_node:main',
            'vision_node = my_robot.robot_vision:main',
            'start_node = my_robot.robot_start:main',
            #'camera_node = my_robot.vision_real:main',
        ],
    },
)
