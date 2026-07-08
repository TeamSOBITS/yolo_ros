<a name="readme-top"></a>

[EN](README.md) | [JA](README_ja.md)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# YOLO ROS

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li>
      <a href="#setup">Setup</a>
      <ul>
        <li><a href="#environment">Environment</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#parameters">Parameters</a></li>
    <li><a href="#topics">Topics</a></li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Overview
`yolo_ros` is a ROS 2 lifecycle wrapper for Ultralytics YOLO models, supporting YOLO26 (detection, pose, segmentation) and YOLOE (prompt-based detection and segmentation).

**Main Features:**
- Object detection (`Detection2DArray`)
- Human pose estimation (`KeyPointArray`)
- Instance segmentation (`DetectMaskArray`)
- YOLO tracking (BoT-SORT / ByteTrack with `track_id` in `Detection2DArray`)
- Prompt-based detection and segmentation with YOLOE
- Conv+BN layer fusion for faster inference (`fuse`)
- Full ROS 2 lifecycle support (`configure` → `activate` → `deactivate` → `cleanup`)
- All parameters configurable at runtime via `ros2 param set`

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Setup

### Environment

| System | Version |
| ------ | ------- |
| Ubuntu | 24.04 (Noble Numbat) |
| ROS 2  | Jazzy Jalisco |
| Python | 3.12 |

### Installation
1. Move to your ROS 2 `src` directory.
   ```sh
   cd ~/colcon_ws/src/
   ```
2. Clone this repository.
   ```sh
   git clone https://github.com/TeamSOBITS/yolo_ros.git
   ```
3. Navigate into the repository and install dependencies.
   ```sh
   cd yolo_ros
   bash install.sh
   ```
4. Build the package.
   ```sh
   cd ~/colcon_ws/
   colcon build --symlink-install
   source install/setup.bash
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

### Start with launch-time configuration

1. Start in detection mode:
   ```sh
   ros2 launch yolo_ros yolo.launch.py image_topic_name:=/camera/color/image_raw mode:=detect
   ```

2. Start in tracking mode:
   ```sh
   ros2 launch yolo_ros yolo.launch.py image_topic_name:=/camera/color/image_raw mode:=track tracker:=botsort.yaml
   ```

   `tracker` can be `botsort.yaml` or `bytetrack.yaml`.

3. Auto configure and activate are `true` by default at launch time:
   ```sh
   ros2 launch yolo_ros yolo.launch.py auto_configure_2d:=true auto_activate_2d:=true
   ```

4. Enable 3D coordinate pipelines:
   ```sh
   ros2 launch yolo_ros yolo.launch.py use_bbox_to_3d:=true use_keypoint_to_3d:=false use_mask_to_3d:=false
   ```

### Switch after startup with lifecycle / param

1. Manage the lifecycle manually:
   ```sh
   ros2 launch yolo_ros yolo.launch.py auto_configure_2d:=false auto_activate_2d:=false
   ros2 lifecycle set /yolo_node configure
   ros2 lifecycle set /yolo_node activate
   ```

2. Switch models at runtime (`weight_file` and `tracker` require deactivate first):
   ```sh
   ros2 lifecycle set /yolo_node deactivate
   ros2 param set /yolo_node weight_file yolo26n-pose.pt
   ros2 lifecycle set /yolo_node activate
   ```

3. Switch to tracking mode:
   ```sh
   ros2 lifecycle set /yolo_node deactivate
   ros2 param set /yolo_node yolo_mode track
   ros2 param set /yolo_node tracker botsort.yaml
   ros2 lifecycle set /yolo_node activate
   ```

4. Switch back to detection mode:
   ```sh
   ros2 param set /yolo_node yolo_mode detect
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Parameters

The following parameters can be set via the launch file or `ros2 param set` at runtime.

`mode:=detect|track` is the launch-time mode selector. `mode:=detect` uses `predict()` and `mode:=track` uses `track()`. Inside the node, the same meaning is exposed as the `yolo_mode` parameter, so you can switch with `ros2 param set /yolo_node yolo_mode track` or `detect`. `tracker` is the tracker selection used only in tracking mode, and can be `botsort.yaml` or `bytetrack.yaml`.

| Parameter              | Description                                                                                     | Default                        | Runtime update  |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------ | --------------- |
| `weight_file`          | YOLO weight filename                                                                            | `yolo26n.pt`                   | inactive only   |
| `weights_path`         | Directory containing the weight file                                                            | `<package>/weights`            | inactive only   |
| `conf`                 | Detection confidence threshold (0.0, 1.0]                                                       | `0.5`                         | yes             |
| `iou`                  | NMS IoU threshold (0.0, 1.0]                                                                    | `0.7`                          | yes             |
| `yolo_mode`            | Node execution mode. `detect` uses `predict()` and `track` uses `track()`                      | `detect`                       | yes             |
| `tracker`              | Ultralytics tracker config used in tracking mode (`botsort.yaml` or `bytetrack.yaml`)          | `botsort.yaml`                 | inactive only   |
| `use_tracking`         | Legacy compatibility parameter. `true` maps to `track`, `false` maps to `detect`              | `false`                        | yes             |
| `use_detection_filter` | Whether to use `filter_classes` provided from `config/detection_filters.yaml` or elsewhere     | `true`                         | yes             |
| `filter_classes`       | Class names to keep for YOLO detection, visualization, and tracking (empty or only empty strings = all classes) | `['person', 'bottle']` | yes             |
| `keypoint_name_list`   | Keypoint names for pose models (positional, max = model keypoint count)                         | `['']`                         | yes             |
| `yoloe_prompts`        | Text prompts for YOLOE models (required when using YOLOE)                                       | `['']`                         | yes             |
| `image_reliability`    | QoS reliability for the image subscription (`best_effort`, `reliable`, `system_default`, ...)  | `best_effort`                  | inactive only   |
| `device`               | Inference device (`cuda`, `cpu`, `cuda:0`, ...)                                                 | `cuda` if available else `cpu` | inactive only   |
| `fuse`                 | Fuse Conv+BN layers after load for faster inference                                             | `true`                         | inactive only   |
| `auto_configure_2d`    | Configure the YOLO lifecycle node on startup                                                    | `true`                         | —               |
| `auto_activate_2d`     | Activate the YOLO lifecycle node on startup                                                     | `true`                         | —               |
| `auto_configure_3d`    | Configure the image_to_position lifecycle node on startup                                       | `false`                        | —               |
| `auto_activate_3d`     | Activate the image_to_position lifecycle node on startup                                        | `false`                        | —               |
| `use_bbox_to_3d`       | Launch the `bbox_to_3d` 3D detection pipeline                                                   | `false`                        | —               |
| `use_keypoint_to_3d`   | Launch the `keypoint_to_3d` 3D pipeline                                                         | `false`                        | —               |
| `use_mask_to_3d`       | Launch the `mask_to_3d` 3D pipeline                                                             | `false`                        | —               |

> **Note:** `weight_file`, `weights_path`, `device`, `fuse`, `image_reliability`, and `tracker` require the node to be `inactive` (deactivated) before changing.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Topics

### Publications

| Topic                     | Type                                  | Description                          |
| ------------------------- | ------------------------------------- | ------------------------------------ |
| `<node>/detected_image`   | `sensor_msgs/Image`                   | Annotated visualization image        |
| `<node>/object_boxes`     | `vision_msgs/Detection2DArray`        | Bounding boxes per detected instance |
| `<node>/object_keypoints` | `sobits_interfaces/KeyPointArray`     | Keypoints (pose models only)         |
| `<node>/object_masks`     | `sobits_interfaces/DetectMaskArray`   | Instance masks (segmentation models) |

### Subscriptions

| Topic                | Type                    | Description        |
| -------------------- | ----------------------- | ------------------ |
| `<image_topic_name>` | `sensor_msgs/Image`     | Input camera image |

> `<node>` defaults to `yolo_node`. Override with the `node_name` launch argument. When tracking is enabled, `Detection2D.id` and `DetectMask.instance_id` contain `class_name:track_id`, such as `person:1`. The class name is also kept in `Detection2D.results[].hypothesis.class_id`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Demo

| Object Detection | Pose Estimation | Segmentation |
|:---:|:---:|:---:|
| ![](docs/yolo26n.jpg) | ![](docs/yolo26n-pose.jpg) | ![](docs/yoloe-26n-seg.jpg) |

### Detection
```bash
ros2 lifecycle set /yolo_node deactivate
ros2 param set /yolo_node weight_file yolo26n.pt
ros2 param set /yolo_node filter_classes "['person', 'laptop']"
ros2 lifecycle set /yolo_node activate
```

### Pose Estimation
```bash
ros2 lifecycle set /yolo_node deactivate
ros2 param set /yolo_node weight_file yolo26n-pose.pt
ros2 param set /yolo_node keypoint_name_list "['nose', 'left_eye', 'right_eye']"
ros2 lifecycle set /yolo_node activate
```

### Segmentation
```bash
ros2 lifecycle set /yolo_node deactivate
ros2 param set /yolo_node weight_file yolo26n-seg.pt
ros2 lifecycle set /yolo_node activate
```

### YOLOE Segmentation
```bash
ros2 lifecycle set /yolo_node deactivate
ros2 param set /yolo_node weight_file yoloe-26n-seg.pt
ros2 param set /yolo_node yoloe_prompts "['bottle', 'laptop']"
ros2 lifecycle set /yolo_node activate
```

### YOLO Tracking
Start directly in tracking mode:
```bash
ros2 launch yolo_ros yolo.launch.py mode:=track tracker:=botsort.yaml
```

To use ByteTrack:
```bash
ros2 launch yolo_ros yolo.launch.py mode:=track tracker:=bytetrack.yaml
```

To switch into tracking mode after startup:
```bash
ros2 lifecycle set /yolo_node deactivate
ros2 param set /yolo_node yolo_mode track
ros2 param set /yolo_node tracker botsort.yaml
ros2 lifecycle set /yolo_node activate
```

To track only people:
```bash
ros2 param set /yolo_node filter_classes "['person']"
```

To track multiple classes:
```bash
ros2 param set /yolo_node filter_classes "['person', 'car']"
```

To temporarily return to all classes:
```bash
ros2 param set /yolo_node use_detection_filter false
```

`filter_classes` can be kept in `config/detection_filters.yaml`. When `use_detection_filter:=true`, class names are resolved through the model `names` and passed to `predict()` / `track()` as the YOLO `classes` argument. When `use_detection_filter:=false`, `filter_classes` is ignored and all model classes are used.

The `track_id` stored in `Detection2D.id` and `DetectMask.instance_id` is a short-term tracking ID while the same object remains continuously visible. The same physical object may receive a new `track_id` after deactivate / activate, changing `tracker`, changing `filter_classes`, changing `use_detection_filter`, or disappearing from view long enough to be reacquired later. YOLO tracking is not a re-identification system, so persistent identity across those events is not guaranteed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Model Downloads

Download the `.pt` file for your task and place it in the [weights](./weights/) directory:

| Task | Model page |
| ---- | ---------- |
| Detection | [YOLO26 Detection Models](https://docs.ultralytics.com/tasks/detect#models) |
| Segmentation | [YOLO26 Segmentation Models](https://docs.ultralytics.com/tasks/segment#models) |
| Semantic Segmentation | [YOLO26 Semantic Models](https://docs.ultralytics.com/tasks/semantic#models) |
| Pose Estimation | [YOLO26 Pose Models](https://docs.ultralytics.com/tasks/pose#models) |
| YOLOE (open-vocabulary) | [YOLOE-26 Models](https://docs.ultralytics.com/models/yolo26#yoloe-26-open-vocabulary-instance-segmentation) |

Then set the filename via the `weight_file` launch argument or at runtime:
```sh
ros2 param set /yolo_node weight_file yolo26n.pt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


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
