import os
from glob import glob
from setuptools import setup

package_name = "yolo_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
        (os.path.join('share', package_name, 'yolo_world_classes'), glob('yolo_world_classes/*')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Miguel Ángel González Santamarta",
    maintainer_email="mgons@unileon.es",
    description="YOLO for ROS 2",
    license="GPL-3",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_ros.yolo_node:main",
            # "tracking_node = yolo_ros.tracking_node:main",
        ],
    },
)
