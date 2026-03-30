<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# Yolo ROS

<details>
  <summary>目次</summary>
  <ol>
    <li>
      <a href="#概要">概要</a>
    </li>
    <li>
      <a href="#対応モデル">対応モデル</a>
    </li>
    <li>
      <a href="#セットアップ">セットアップ</a>
      <ul>
        <li><a href="#環境条件">環境条件</a></li>
        <li><a href="#インストール方法">インストール方法</a></li>
      </ul>
    </li>
    <li><a href="#実行操作方法">実行・操作方法</a></li>
    <li><a href="#パラメーター">パラメーター</a></li>
    <li><a href="#入出力">入出力</a></li>
    <li><a href="#ライフサイクルノード">ライフサイクルノード</a></li>
    <li><a href="#デモ">デモ</a></li>
     <li><a href="#マイルストーン">マイルストーン</a></li>
    <li><a href="#参考文献">参考文献</a></li>
  </ol>
</details>

## 概要
yolo_rosは，UltralyticsのYOLOモデル（YOLOv3からYOLOv11，YOLO-NAS，YOLOEなど）をROS 2で利用するためのラッパーです．これにより，以下の機能がROS 2環境で実現できます．

- 物体検出 (Object Detection)
- 人間の姿勢推定 (Human Pose Estimation)
- インスタンスセグメンテーション (Instance Segmentation)
- YOLOEによる効率的な推論とプロンプトベースの検出

## セットアップ
本レポジトリのセットアップ方法について説明します．

### 環境条件

| System  | Version |
| ------------- | ------------- |
| Ubuntu | 24.04 (Noble Numbat) |
| ROS | Jazzy Jalisco |
| Python | 3.10~ |

### インストール方法
1. ROS 2の`src`フォルダに移動します．
   ```sh
   cd ~/colcon_ws/src/
   ```
2. 本レポジトリをcloneします．
   ```sh
   git clone -b jazzy-devel https://github.com/TeamSOBITS/yolo_ros.git
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

## 実行・操作方法
1. カメラを起動し，[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**image_topic_name**を使用するカメラのトピック名に書き換える．

    ```sh
    ros2 launch yolo_ros yolo.launch.py
    ```

2. カスタムの重みファイルを使用する場合：
    - 用意した`.pt`ファイルを`weights`に配置してください．

3. 3D座標変換（物体位置の推定）を有効にする場合：
    ```sh
    ros2 launch yolo_ros yolo.launch.py use_3d:=True
    ```



## パラメーター
以下は[yolo.launch.py](launch/yolo.launch.py)およびノードで設定可能な主なパラメーターです．
詳細は[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/#inference-arguments)も参照してください．

| パラメーター名            | 型            | 説明                               |
| ------------------ | ------------ | -------------------------------- |
| image_topic_name   | string       | 入力となる画像トピック名                     |
| weight_file        | string       | 使用する重みファイル名（weightsフォルダ内）        |
| weights_path       | string       | 重みファイルが保存されているディレクトリパス           |
| execute_default    | bool         | 起動時に自動でノードをConfigure/Activateするか |
| conf               | double       | 検出の信頼度しきい値                       |
| iou                | double       | NMSのIoUしきい値                      |
| filter_classes     | string_array | 検出対象を絞り込むクラス名のリスト                |
| keypoint_name_list | string_array | 姿勢推定時のキーポイント名のリスト                |
| yoloe_prompts      | string_array | YOLOEで使用する検出プロンプト                |

## デモ
| 物体検出 | 姿勢推定 | インスタンスセグメンテーション |
|:---:|:---:|:---:|
| ![](docs/yolo26n.jpg) | ![](docs/yolo26n-pose.jpg) | ![](docs/yoloe-26n-seg.jpg) |



## 参考文献
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
