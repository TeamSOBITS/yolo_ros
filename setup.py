import os
from glob import glob
from setuptools import setup

package_name = "yolo_ros"

setup(
    name=package_name,
    version="4.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "weights"), glob("weights/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sobits",
    maintainer_email="yasumasashige790@gmail.com",
    description="YOLO for ROS 2",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_ros.yolo_node:main",
        ],
    },
)
