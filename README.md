<a name="readme-top"></a>

[EN](README.md) | [JA](README_ja.md)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# Yolo ROS

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#supported-models">Supported Models</a>
    </li>
    <li>
      <a href="#setup">Setup</a>
      <ul>
        <li><a href="#environment">Environment</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#parameters">Parameters</a></li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Overview
`yolo_ros` is a wrapper package for using Ultralytics YOLO models in ROS 2, including YOLOv3 through YOLOv11, YOLO-NAS, and YOLOE.

It provides the following features in a ROS 2 environment:

- Object detection
- Human pose estimation
- Instance segmentation
- Efficient inference and prompt-based detection with YOLOE

## Setup
This section explains how to set up this repository.

### Environment

| System  | Version |
| ------------- | ------------- |
| Ubuntu | 24.04 (Noble Numbat) |
| ROS | Jazzy Jalisco |
| Python | 3.10~ |

### Installation
1. Move to your ROS 2 `src` directory.
   ```sh
   cd ~/colcon_ws/src/
   ```
2. Clone this repository.
   ```sh
   git clone -b jazzy-devel https://github.com/TeamSOBITS/yolo_ros.git
   ```
3. Move into the repository.
   ```sh
   cd yolo_ros
   ```
4. Install dependencies.
    ```sh
    bash install.sh
    ```
5. Build the package.
   ```sh
   cd ~/colcon_ws/
   colcon build --symlink-install
   ```

## Usage
1. Launch your camera and update **image_topic_name** in [yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/jazzy-devel/launch/yolo.launch.py) to match your camera topic.

    ```sh
    ros2 launch yolo_ros yolo.launch.py
    ```

2. If you want to use a custom weight file:
   - Place your `.pt` file in the `weights` directory.

3. If you want to use 3D coordinate conversion for object position estimation, enable the required 3D pipelines with launch arguments.

    Example: enable only `bbox_to_3d`
    ```sh
    ros2 launch yolo_ros yolo.launch.py \
      use_bbox_to_3d:=True \
      use_keypoint_to_3d:=False \
      use_mask_to_3d:=False
    ```

    Example: enable both `bbox_to_3d` and `keypoint_to_3d`
    ```sh
    ros2 launch yolo_ros yolo.launch.py \
      use_bbox_to_3d:=True \
      use_keypoint_to_3d:=True \
      use_mask_to_3d:=False
    ```

4. If you want to control YOLO lifecycle startup transitions independently:
    ```sh
    ros2 launch yolo_ros yolo.launch.py \
      auto_configure_2d:=True \
      auto_activate_2d:=False
    ```

## Parameters
The following are the main parameters available in [yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/jazzy-devel/launch/yolo.launch.py) and the node itself.
See [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/#inference-arguments) for additional inference arguments.

| Parameter          | Type         | Description                        |
| ------------------ | ------------ | -------------------------------- |
| image_topic_name   | string       | Input image topic name             |
| weight_file        | string       | Weight file name in the `weights` directory |
| weights_path       | string       | Directory path where weight files are stored |
| auto_configure_2d  | bool         | Whether to configure the YOLO lifecycle node on startup |
| auto_activate_2d   | bool         | Whether to activate the YOLO lifecycle node on startup |
| auto_configure_3d  | bool         | Whether to configure the Image to Position lifecycle node on startup |
| auto_activate_3d   | bool         | Whether to activate the Image to Position lifecycle node on startup |
| conf               | double       | Detection confidence threshold     |
| iou                | double       | NMS IoU threshold                  |
| use_bbox_to_3d     | bool         | Whether to launch `bbox_to_3d`     |
| use_keypoint_to_3d | bool         | Whether to launch `keypoint_to_3d` |
| use_mask_to_3d     | bool         | Whether to launch `mask_to_3d`     |
| filter_classes     | string_array | List of class names to filter detections |
| keypoint_name_list | string_array | List of keypoint names for pose estimation |
| yoloe_prompts      | string_array | Detection prompts used by YOLOE    |

## Demo
| Object Detection | Pose Estimation | Segmentation |
|:---:|:---:|:---:|
| ![](docs/yolo26n.jpg) | ![](docs/yolo26n-pose.jpg) | ![](docs/yoloe-26n-seg.jpg) |

### Detection
  ```bash
  ros2 param set /yolo_ros weight_file "yolo26n.pt"
  ros2 param set /yolo_ros filter_classes "['person', 'laptop']"
  ros2 param set /yolo_ros filter_classes "['']"
  ```

### Pose
```bash
ros2 param set /yolo_ros weight_file "yolo26n-pose.pt"
```

### Segmentation
```bash
ros2 param set /yolo_ros weight_file "yolo26n-seg.pt"
ros2 param set /yolo_ros filter_classes "['person', 'laptop']"
ros2 param set /yolo_ros filter_classes "['']"
```

### YOLOE Segmentation
```bash
ros2 param set /yolo_ros weight_file "yoloe-26n-seg.pt"
ros2 param set /yolo_ros yoloe_prompts "['bottle', 'laptop']"
```

## References
* [Ultralytics Documentation](https://docs.ultralytics.com/)

[contributors-shield]: https://img.shields.io/github/contributors/TeamSOBITS/yolo_ros.svg?style=for-the-badge
[contributors-url]: https://github.com/TeamSOBITS/yolo_ros/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TeamSOBITS/yolo_ros.svg?style=for-the-badge
[forks-url]: https://github.com/TeamSOBITS/yolo_ros/network/members
[stars-shield]: https://img.shields.io/github/stars/TeamSOBITS/yolo_ros.svg?style=for-the-badge
[stars-url]: https://github.com/TeamSOBITS/yolo_ros/stargazers
[issues-shield]: https://img.shields.io/github/issues/TeamSOBITS/yolo_ros.svg?style=for-the-badge
[issues-url]: https://github.com/TeamSOBITS/yolo_ros/issues
[license-shield]: https://img.shields.io/github/license/TeamSOBITS/yolo_ros.svg?style=for-the-badge
[license-url]: LICENSE
