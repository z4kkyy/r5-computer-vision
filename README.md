# r5-computer-vision

This repository contains a gameplay assistance tool that utilizes advanced computer vision technology. It combines precise object detection using YOLOv8 and TensorRT Engine with smooth mouse control using PID control to assist in aim training against dummies (bots) in R5Reloaded, an SDK for creating mods for Apex Legends.

In the following video, the mouse is not being moved at all.

https://github.com/user-attachments/assets/fe0df495-b5f1-41ee-aca0-3e668264a7f6

## Disclaimer

This software is designed to control mouse input and does not read or write to game binaries. Its use should be limited to aim training against dummies (bots) in R5Reloaded, an SDK for creating mods for Apex Legends. Using it against human players online is ethically problematic and carries a high risk of account banning, thus it is not recommended.

Furthermore, this software cannot be used with the official Apex Legends, as the current implementation fundamentally cannot distinguish between enemies and allies.

## Key Features

- Real-time object detection using YOLOv8 and TensorRT Engine: Achieves high-speed detection in tens of milliseconds and high-precision target locking
- Non-invasive screen capture technology: Utilizes DirectX to capture frames with ultra-low latency of less than 1ms
- Precise mouse control using PID control: Achieves natural and smooth aiming movements, replicating human-like behavior

## Version Requirements

| Component | Version |
| :-------: | :-----: |
| CUDA      | 12.1    |
| cuDNN     | 8.9.0   |
| TensorRT  | 10.12.0 |
| PyTorch   | 2.5.1   |

## Main Features
**Operation Flow**: This program runs the following sequence in a loop by executing the `main.py` script:
  - Start a thread that asynchronously updates key states. (Line 81 in `main.py`)
  - For each main loop iteration:
    - Read key and mouse states. (Line 111 in `main.py`)
    - Execute the execute method of the r5CVCore class (Line 118 in `main.py`). Within this:
      - Get the latest frame (screenshot) from the DXCamera instance (less than 1ms).
      - Perform object detection.
      - Display the image with bounding boxes (optional).
      - Calculate conditions and move the mouse cursor appropriately if met.

**Usage**
- Shift
    - Press and hold the shift key to **lock onto a target**.
- Toggle 1
    - This feature **continuously locks onto targets**, and should **not be used while firing**.
    - Toggle by pressing the `'y'` key. The key can be changed in config.yaml.
- Toggle 2 (ADS needs to be changed from **Toggle** to **Hold**.)
    - This feature **continuously locks onto targets while scoped (ADS)**.
    - Toggle by pressing the `'u'` key. The key can be changed in config.yaml.
- HOME
    - Press the home key to terminate all processes.

## Updating Settings
Various parameters can be adjusted in `config.yaml`.


---

# r5-computer-vision

本レポジトリはCV（コンピュータービジョン）技術を活用したゲームプレイ支援ツールです。YOLOv8およびTensorRT Engineによる精密なオブジェクト検出とPID制御を用いた滑らかなマウス制御を組み合わせ、Apex LegendsのMOD作成用のSDKであるR5Reloadedでのダミー（ボット）に対するエイムトレーニングを支援するものです。

## 注意事項

このソフトウェアは、マウス入力の制御を目的として設計されており、ゲームのバイナリの読み書きは一切行いません。使用については、Apex LegendsのMOD作成用のSDKであるR5Reloadedでのダミー（ボット）に対するエイムトレーニング限定してください。オンラインにおける人間のプレイヤーに対する使用は倫理的に問題があり、アカウントのバンのリスクが高いため、推奨されません。

さらに、このソフトウェアは公式のApex Legendsでは使用できません。現在の実装では、敵と味方を区別することが根本的に不可能であるためです。

## 主な特徴

- YOLOv8およびTensorRT Engineを使用したリアルタイムオブジェクト検出：数十ミリ秒単位の高速な検出と高精度なターゲットロックを実現
- 非侵襲的なスクリーンキャプチャ技術：DirectXを利用し、1ms未満の超低遅延でフレームを取得
- PID制御を用いた精密なマウス制御：自然で滑らかなエイム動作を実現し、人間の動きに近い挙動を再現

## バージョン要件

| コンポーネント | バージョン |
| :-----------: | :-------: |
| CUDA          | 12.1      |
| cuDNN         | 8.9.0     |
| TensorRT      | 10.12.0   |
| PyTorch       | 2.5.1     |

## 主要機能
**動作フロー**：このプログラムは`main.py`スクリプトを実行することで、以下のシーケンスをループします：
  - キーの状態を非同期で更新するスレッドを開始します。（`main.py`、81行目）
  - 各メインループの反復で：
    - キーとマウスの状態を読み取る。（`main.py`、111行目）
    - r5CVCoreクラスのexecuteメソッドを実行する（`main.py`、118行目）。その中では：
      - DXCameraインスタンスから最新のフレーム（スクリーンショット）を取得（1ms未満）。
      - オブジェクト検出の実行。
      - バウンディングボックス付きの画像を表示（オプション）。
      - 条件を計算し、満たされた場合に適切にマウスカーソルを移動。

**使用方法**
- シフト
    - シフトキーを押し続けて**標的をロック**します。
- トグル1
    - この機能は**ターゲットをロックし続ける**もので、**発射中は使用しません**。
    - `'y'`キーを押してトグルします。config.yamlでキーを変更できます。
- トグル2（ADSを**トグル**から**ホールド**に変更する必要があります。）
    - この機能は**スコープ中（ADS中）にターゲットをロックし続ける**ものです。
    - `'u'`キーを押してトグルします。config.yamlでキーを変更できます。
- HOME
    - ホームキーを押して全プロセスを終了します。

## 設定の更新
`config.yaml`において各種パラメーターの調整が可能です。
