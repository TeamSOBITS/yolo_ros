<a name="readme-top"></a>

[JP](README.md) | [EN](README_en.md)

# yolo_ros

目次
1. [概要](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/README.md#概要)
2. [対応モデル](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/README.md#対応モデル)
3. [セットアップ](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/README.md#セットアップ)
4. [実行・操作方法](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/README.md#実行・操作方法)
5. [パラメーター](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/README.md#パラメーター)

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

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 対応モデル
|  |  |  |
| ------------- | ------------- | ------------- |
| [YOLOv3](https://docs.ultralytics.com/models/yolov3/) | [YOLOv4](https://docs.ultralytics.com/models/yolov4/) | [YOLOv5](https://docs.ultralytics.com/models/yolov5/) |
| [YOLOv6](https://docs.ultralytics.com/models/yolov6/) | [YOLOv7](https://docs.ultralytics.com/models/yolov7/) | [YOLOv8](https://docs.ultralytics.com/models/yolov8/) |
| [YOLOv9](https://docs.ultralytics.com/models/yolov9/) | [YOLOv10](https://docs.ultralytics.com/models/yolov10/) | [YOLOv11](https://docs.ultralytics.com/models/yolo11/) |
| [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) | [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) |  |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

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
以下は[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)で設定できるパラメーターである。
詳細は[Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments)で確認できる。

| パラメーター名  | 説明 | デフォルト値 |
| ------------- | ------------- | ------------- |
| model_type | Ultralyticsモデルタイプ | YOLO |
| init_prediction | 起動時に検出の推論を行うかどうか | True |
| image_show | 検出結果の画像を表示するかどうか | False |
| threshold | 検出の閾値 | 0.5 |
| iou | 検出した位置がどれだけ正解と重なっているか(低いと同じ位置に複数のbboxが出現) | 0.7 |
| imgsz_height, imgsz_width | 推論のための画像高さ/幅 | 480/640 |
| half | 半精度 (FP16) 推論を有効にするか | False |
| max_det | 画像あたりの最大検出数 | 300 |
| agnostic_nms | クラスに依存しないNMSを有効にするか | False |
| retina_masks | 高解像度セグメンテーションマスクを使用するか | False |
| use_3d | 3D検出を有効にするか | True |
| cluster_tolerance |  | 0.01 |
| min_clusterSize |  | 100 |
| max_clusterSize |  | 20000 |
| noise_point_cloud_range |  | 0.01 |
| fast_shot | fast_shotを有効にするかどうか(bboxが大きいときに検出速度が上昇) | true |
| enable_id | 検出した物体のラベルの後ろにIDをつけるかどうか(例:apple_01) | false |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## ライフサイクルノード (Lifecycle Nodes)
yolo_rosのすべてのノードはライフサイクルノードをサポートしています。これにより、未設定 (unconfigured) および非アクティブ (inactive) 状態での負荷を軽減し、アクティブ (active) 状態でのみモデルのロードとサブスクライバーのアクティブ化を行います。

状態ごとのリソース比較 (yolov8m.ptモデル、30fpsビデオストリーム)

| State    | CPU Usage (i7 12th Gen) | VRAM Usage | Bandwidth Usage |
| -------- | ----------------------- | ---------- | --------------- |
| Active   | 40-50% in one core      | 628 MB     | Up to 200 Mbps  |
| Inactive | ~5-7% in one core       | 338 MB     | 0-20 Kbps       |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## デモ
## 物体検出 (Object Detection)（トラッキングを含む標準動作）

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## インスタンスセグメンテーション (Instance Segmentation)（yolov8m-seg.ptモデルを使用）

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## 人間の姿勢推定 (Human Pose)（yolov8m-pose.ptモデルを使用）

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## 3D物体検出 (3D Object Detection)

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D物体検出 (インスタンスセグメンテーションマスクを使用)

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D人間の姿勢推定 (3D Human Pose)

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>
