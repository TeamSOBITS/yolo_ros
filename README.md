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
      <a href="#セットアップ">セットアップ</a>
      <ul>
        <li><a href="#環境条件">環境条件</a></li>
        <li><a href="#インストール方法">インストール方法</a></li>
      </ul>
    </li>
    <li><a href="#実行操作方法">実行・操作方法</a></li>
    <li><a href="#パラメーター">パラメーター</a></li>
    <li><a href="#run-control">Run Control</a></li>
    <li><a href="#参考文献">参考文献</a></li>
  </ol>
</details>

## 概要
YOLO ROS は，UltralyticsのYOLOモデルをROSで利用するためのラッパーです．これにより，以下の機能がROS環境で実現できます．

- 物体検出 (Object Detection)

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## セットアップ
本レポジトリのセットアップ方法について説明します．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 環境条件

| System  | Version |
| ------------- | ------------- |
| Ubuntu | 20.04 (Focal Fossa) |
| ROS | Noetic Ninjemys |
| Python | 3.8 |


<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### インストール方法
1. ROSの`src`フォルダに移動します．
   ```sh
   cd　~/catkin_ws/src/
   ```
2. 本レポジトリをcloneします．
   ```sh
   git clone -b noetic-devel https://github.com/TeamSOBITS/yolo_ros.git
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
   cd ~/catkin_ws/
   catkin_make
   ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- 実行・操作方法 -->
## 実行・操作方法
1. カメラを起動し，[yolo.launch.xml](launch/yolo.launch.xml)の**image_topic_name**を使用するカメラのトピック名に書き換える．
   
   例
   ```sh
   <param name="image_topic_name" value="/camera_main/image_raw"/>
   ```

2. ウェイトファイルを設定
    用意したウェイトファイルを[weightsディレクトリ](weights)に入れる．

3. [yolo.launch.xml](launch/yolo.launch.xml)の**model_path**を，手順2で設定したウェイトファイル名に書き換える．
   ```sh
   <param name="model_path" value="$(find yolo_ros)/weights/best.pt" />  ## custom weight file
   ```
5. catkin_makeを実行
   ```sh
   cd ~/catkin_ws/
   catkin_make
   ```
6. yoloを起動
    ```sh
    roslaunch yolo_ros yolo.launch
    ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## パラメーター
以下は[yolo.launch.xml](launch/yolo.launch.xml)で設定できるパラメーターである．
詳細は[Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments)で確認できる．

| パラメーター名  | 説明 | デフォルト値 |
| ------------- | ------------- | ------------- |
| conf_th | 検出の閾値 | 0.5 |#

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## Run Control

以下のコマンドで YOLO の推論の ON/OFF を切り替え可能．

```sh
# ON
rosservice call /yolo_trigger "data: true"
```

```sh
# OFF
rosservice call /yolo_trigger "data: false"
```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


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

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>
