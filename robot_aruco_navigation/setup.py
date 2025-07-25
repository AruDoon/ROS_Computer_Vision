from setuptools import find_packages, setup

package_name = 'robot_aruco_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_navigation.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='ArUco marker-based robot navigation system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = robot_aruco_navigation.robot_vision_node:main',
            'bridge_node = robot_aruco_navigation.robot_bridge_node:main',
        ],
    },
)