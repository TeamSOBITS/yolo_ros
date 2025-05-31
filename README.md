<a name="readme-top"></a>

[JP](README.md) | [EN](README_en.md)

# yolo_ros

目次
1. [概要](##概要)
2. [対応モデル](##対応モデル)
3. [セットアップ](##セットアップ)
4. [実行・操作方法](##実行・操作方法)
5. [パラメーター](##パラメーター)

## 概要
yolo_rosは、UltralyticsのYOLOモデル（YOLOv3からYOLOv11、YOLO-NAS、YOLO-Worldなど）をROS 2で利用するためのラッパーです。これにより、以下の機能がROS 2環境で実現できます。

- 物体検出 (Object Detection)
- トラッキング (Tracking)
- インスタンスセグメンテーション (Instance Segmentation)
- 人間の姿勢推定 (Human Pose Estimation)
- Oriented Bounding Box (OBB)
- 3D物体検出 (3D Object Detection)：深度画像を使用して3Dバウンディングボックスを生成
- 3Dインスタンスセグメンテーション (3D Instance Segmentation)：深度画像とインスタンスマスクを使用
- 3D人間の姿勢推定 (3D Human Pose Estimation)：深度画像とキーポイントを使用

## 対応モデル
- [YOLOv3](https://docs.ultralytics.com/models/yolov3/)
- [YOLOv4](https://docs.ultralytics.com/models/yolov4/)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/)
- [YOLOv6](https://docs.ultralytics.com/models/yolov6/)
- [YOLOv7](https://docs.ultralytics.com/models/yolov7/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)

## セットアップ
本レポジトリのセットアップ方法について説明します．

### 環境条件

| System  | Version |
| ------------- | ------------- |
| Ubuntu | 22.04 (Jammy Jellyfish) |
| ROS | Humble Hawksbill |
| Python | 3.0~ |

### インストール方法
1. ROS2の`src`フォルダに移動します．
   ```sh
   cd　~/colcon_ws/src/
   ```
2. 本レポジトリをcloneします．
   ```sh
   git clone -b humble-devel https://github.com/TeamSOBITS/yolo_ros.git
   ```
3. レポジトリの中へ移動します．
   ```sh
   cd yolo_ros
   ```
4. 依存パッケージをインストールします．
    ```sh
    bash install.sh
    ```
5. パッケージをコンパイルします．
   ```sh
   cd ~/colcon_ws/
   colcon build --symlink-install
   ```
<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


<!-- 実行・操作方法 -->
## 実行・操作方法
1. カメラを起動し、[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**image_topic_name**を使用するカメラのトピック名に書き換える。
   
   例
   ```sh
   default_value="/camera/color/image_raw",          ## orbbec_series
   ```
2. RGBDカメラを使用する場合は、[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**point_cloud_topic**も使用するカメラの点群のトピック名に書き換える。
   
   例
   ```sh
   default_value="/camera/depth_registered/points",     ## orbbec_series
   ```. 
3. ウェイトファイルを設定\
    用意したウェイトファイルを[weightsディレクトリ](https://github.com/TeamSOBITS/yolo_ros/tree/humble-devel/weights)に入れる。
4. [yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**weight_file**を、手順3で設定したウェイトファイル名に書き換える。
   ```sh
   default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt"),  ## custom weight file
   ```
5. colcon buildを実行
   ```sh
   cd ~/colcon_ws/
   colcon build --symlink-install
   ```
6. yoloを起動
    ```sh
    ros2 launch yolo_ros yolo.launch.py
    ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## パラメーター
- init_prediction
- image_show
- threshold
  - 検出閾値
- iou
  - 非最大抑制 (NMS) のためのIoU閾値
- use_3d
  - 3D検出を有効にするか
- fast_shot
- enable_id

These are the parameters from the [yolo.launch.py](./yolo_bringup/launch/yolo.launch.py), used to launch all models. Check out the [Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments) for more details.

- **model_type**: Ultralytics model type (default: YOLO)
- **model**: YOLO model (default: yolov8m.pt)
- **tracker**: tracker file (default: bytetrack.yaml)
- **device**: GPU/CUDA (default: cuda:0)
- **enable**: whether to start YOLO enabled (default: True)
- **threshold**: detection threshold (default: 0.5)
- **iou**: intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS) (default: 0.7)
- **imgsz_height**: image height for inference (default: 480)
- **imgsz_width**: image width for inference (default: 640)
- **half**: whether to enable half-precision (FP16) inference speeding up model inference with minimal impact on accuracy (default: False)
- **max_det**: maximum number of detections allowed per image (default: 300)
- **augment**: whether to enable test-time augmentation (TTA) for predictions improving detection robustness at the cost of speed (default: False)
- **agnostic_nms**: whether to enable class-agnostic Non-Maximum Suppression (NMS) merging overlapping boxes of different classes (default: False)
- **retina_masks**: whether to use high-resolution segmentation masks if available in the model, enhancing mask quality for segmentation (default: False)
- **input_image_topic**: camera topic of RGB images (default: /camera/rgb/image_raw)
- **image_reliability**: reliability for the image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **input_depth_topic**: camera topic of depth images (default: /camera/depth/image_raw)
- **depth_image_reliability**: reliability for the depth image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **input_depth_info_topic**: camera topic for info data (default: /camera/depth/camera_info)
- **depth_info_reliability**: reliability for the depth info topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **target_frame**: frame to transform the 3D boxes (default: base_link)
- **depth_image_units_divisor**: divisor to convert the depth image into meters (default: 1000)
- **maximum_detection_threshold**: maximum detection threshold in the z-axis (default: 0.3)
- **use_tracking**: whether to activate tracking after detection (default: True)
- **use_3d**: whether to activate 3D detections (default: False)
- **use_debug**: whether to activate debug node (default: True)

## Lifecycle Nodes

Previous updates add Lifecycle Nodes support to all the nodes available in the package.
This implementation tries to reduce the workload in the unconfigured and inactive states by only loading the models and activating the subscriber on the active state.

These are some resource comparisons using the default yolov8m.pt model on a 30fps video stream.

| State    | CPU Usage (i7 12th Gen) | VRAM Usage | Bandwidth Usage |
| -------- | ----------------------- | ---------- | --------------- |
| Active   | 40-50% in one core      | 628 MB     | Up to 200 Mbps  |
| Inactive | ~5-7% in one core       | 338 MB     | 0-20 Kbps       |

## Demos

## Object Detection

This is the standard behavior of yolo_ros which includes object tracking.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## Instance Segmentation

Instance masks are the borders of the detected objects, not all the pixels inside the masks.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## Human Pose

Online persons are detected along with their keypoints.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## 3D Object Detection

The 3D bounding boxes are calculated by filtering the depth image data from an RGB-D camera using the 2D bounding box. Only objects with a 3D bounding box are visualized in the 2D image.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

In this, the depth image data is filtered using the max and min values obtained from the instance masks. Only objects with a 3D bounding box are visualized in the 2D image.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D Human Pose

Each keypoint is projected in the depth image and visualized using purple spheres. Only objects with a 3D bounding box are visualized in the 2D image.

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)
