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
    <li><a href="#ライフサイクルノード">ライフサイクルノード</a></li>
    <li><a href="#デモ">デモ</a></li>
     <li><a href="#マイルストーン">マイルストーン</a></li>
    <li><a href="#参考文献">参考文献</a></li>
  </ol>
</details>

## 概要
yolo_rosは，UltralyticsのYOLOモデル（YOLOv3からYOLOv11，YOLO-NAS，YOLO-Worldなど）をROS 2で利用するためのラッパーです．これにより，以下の機能がROS2環境で実現できます．

- 物体検出 (Object Detection)
- トラッキング (Tracking)
- インスタンスセグメンテーション (Instance Segmentation)
- 人間の姿勢推定 (Human Pose Estimation)



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



<!-- 実行・操作方法 -->
## 実行・操作方法
1. カメラを起動し，[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**image_topic_name**を使用するカメラのトピック名に書き換える．
   
   例
   ```sh
   default_value="/camera/color/image_raw"          # orbbec_series
   ```
2. RGBDカメラを使用する場合は，[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**point_cloud_topic**も使用するカメラの点群のトピック名に書き換える．
   
   例
   ```sh
   default_value="/camera/depth_registered/points"     # orbbec_series
   ```
3. ウェイトファイルを設定\
    用意したウェイトファイルを[weightsディレクトリ](https://github.com/TeamSOBITS/yolo_ros/tree/humble-devel/weights)に入れる．
4. [yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)の**weight_file**を，手順3で設定したウェイトファイル名に書き換える．
   ```sh
   default_value=os.path.join(get_package_share_directory("yolo_ros"), "weights", "best.pt")  ## custom weight file
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



## パラメーター
以下は[yolo.launch.py](https://github.com/TeamSOBITS/yolo_ros/blob/humble-devel/launch/yolo.launch.py)で設定できるパラメーターである．
詳細は[Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments)で確認できる．

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
| cluster_tolerance | どの程度離れた点群までは同一の物体とみなすかのしきい値．BoundingBox内に点群を飛ばした場合に，対象物に点群があたり，しきい値いないにある点群を1物体とみなしクラス分けを行います． そのため，あまり大きくすると点群1つ1つの探索範囲が広がり処理が遅くなってしまいます． | 0.01 |
| min_clusterSize | どの程度の数以下の点群の集まりは対象物の点群から棄却するかのしきい値．点群をクラス分けした際に，この数以下の点群数だったらノイズとみなし棄却します． | 100 |
| max_clusterSize | どの程度の数以上の点群の集まりは対象物の点群から棄却するかのしきい値．点群をクラス分けした際に，この数以上の点群数だったら全く別の対象物(物体だったら床の点群など)を捉えてしまったとみなし棄却します． | 20000 |
| noise_point_cloud_range | 対象の物体の点群からノイズ面を除去し，中心座標に近づけるため除去量．クラス分けした点群から物体を抽出した後，床や背後の壁，左右の壁などx,y,z方向に点群をこの値分，更にカットします． こうすることで，より物体の部分のみにかかる点群に絞ることができます． しかし値を大きくしすぎると，物体分の点群まで多く削いでしまうため注意が必用です． | 0.01 |
| fast_shot | fast_shotを有効にするかどうか | true |
| enable_id | 検出した物体のラベルの後ろにIDをつけるかどうか(例: apple_01) | false |


## ライフサイクルノード
yolo_rosのすべてのノードはライフサイクルノードをサポートしています．これにより，未設定 (unconfigured) および非アクティブ (inactive) 状態での負荷を軽減し，アクティブ (active) 状態でのみモデルのロードとサブスクライバーのアクティブ化を行います．

状態ごとのリソース比較 (yolov8m.ptモデル，30fpsビデオストリーム)

| State    | CPU Usage (i7 12th Gen) | VRAM Usage | Bandwidth Usage |
| -------- | ----------------------- | ---------- | --------------- |
| Active   | 40-50% in one core      | 628 MB     | Up to 200 Mbps  |
| Inactive | ~5-7% in one core       | 338 MB     | 0-20 Kbps       |



## デモ
| 物体検出 | 姿勢推定 | インスタンスセグメンテーション |
|:---:|:---:|:---:|
| ![](docs/yolo11n.jpg) | ![](docs/yolo11n-pose.jpg) | ![](docs/yoloe-11s-seg.jpg) |



## 参考文献
* [ultralytics](https://docs.ultralytics.com/ja/)

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
