import os
from glob import glob
from setuptools import find_packages, setup

package_name = "yolo_ros"

setup(
    name=package_name,
    version="4.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "weights"), glob("weights/*")),
    ],
    install_requires=["setuptools", "PySide6"],
    zip_safe=True,
    maintainer="sobits",
    maintainer_email="yasumasashige790@gmail.com",
    description="YOLO for ROS 2",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_ros.yolo_node:main",
            "yolo_gui = yolo_ros.gui.app:main",
            "yolo_camera = yolo_ros.camera_node:main",
        ],
    },
)
