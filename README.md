# r5-computer-vision

This repository contains a gameplay assistance tool that utilizes advanced computer vision technology. It combines precise object detection using YOLOv8 and DXTensorRT Engine with smooth mouse control using PID control to assist in aim training against dummies (bots) in R5Reloaded, an SDK for creating mods for Apex Legends.

In the following video, the mouse is not being moved at all.

https://github.com/user-attachments/assets/fe0df495-b5f1-41ee-aca0-3e668264a7f6

## Disclaimer

This software is designed to control mouse input and does not read or write to game binaries. Its use should be limited to aim training against dummies (bots) in R5Reloaded, an SDK for creating mods for Apex Legends. Using it against human players online is ethically problematic and carries a high risk of account banning, thus it is not recommended.

Furthermore, this software cannot be used with the official Apex Legends, as the current implementation fundamentally cannot distinguish between enemies and allies.

## Key Features

- Real-time object detection using YOLOv8 and TensorRT Engine: Achieves high-speed detection in tens of milliseconds and high-precision target locking
- Non-invasive screen capture technology: Utilizes DirectX to capture frames with ultra-low latency of less than 1ms
- Precise mouse control using PID control: Achieves natural and smooth aiming movements, replicating human-like behavior

## Installation Instructions

**Note:** The inference model in this project maintains compatibility with [https://github.com/Ape-xCV/Apex-CV-YOLO-v8-Aim-Assist-Bot]. If updated models in the original project are functioning correctly, they can be used in this project as well.

# Installation Instructions

## Version Checklist:

| CUDA   | cuDNN | TensorRT | PyTorch |
| :----: | :---: | :------: | :-----: |
| 12.1   | 8.9.0 | 8.6.1.6  | 2.3.0   |

1. Extract `r5-computer-vision.zip` to **C:\temp\r5-computer-vision**.

2. Install `CUDA 12.1.0` from the [`NVIDIA website`](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local).

3. Install `cuDNN 8.9.0`.
   - Register for the [`NVIDIA developer program`](https://developer.nvidia.com/login).
   - Go to the [cuDNN download archive](https://developer.nvidia.com/rdp/cudnn-archive).
   - Click `Download cuDNN v8.9.0 (April 11th, 2023), for CUDA 12.x`.
   - Download `Local Installer for Windows (Zip)`.
   - Unzip `cudnn-windows-x86_64-8.9.0.131_cuda12-archive.zip`.
   - Copy all three folders (`bin`, `include`, `lib`) and paste them into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`.

4. Install `Python 3.11.9 (64-bit)` from the [`Python website`](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe).
   - From custom installation, check "Add Python to environment path".
   - In command prompt, execute the following commands one by one:
   ```shell
   cd /D C:\temp\r5-computer-vision
   python -m pip install --upgrade pip
   python -m venv r5cv
   .\r5cv\Scripts\activate
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. Install `TensorRT`.
   - Go to the [NVIDIA TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download) site.
   - Download `TensorRT 8.6 GA and CUDA 12.0 and 12.1 ZIP Package` from the [NVIDIA website](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip).
   - Extract `TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip` to **C:\TEMP**.
   - After TensorRT is added to PATH, close that Command Prompt and open a new one. Then input:
   ```shell
   cd /D C:\TEMP\TensorRT-8.6.1.6\python
   pip install tensorrt-8.6.1-cp310-none-win_amd64.whl
   ```

6. Export `best_8s.pt` to `best_8s.engine` (if needed):
   - Press **_[Win+R]_** and enter **cmd** to open a Command Prompt. Then input:
   ```shell
   set CUDA_MODULE_LOADING=LAZY
   cd /D C:\TEMP\Ape-xCV\MODEL
   yolo export model=best_8s.pt format=engine opset=12 workspace=7
   ```

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

本レポジトリはCV（コンピュータービジョン）技術を活用したゲームプレイ支援ツールです。YOLOv8およびDXTensorRT Engineによる精密なオブジェクト検出とPID制御を用いた滑らかなマウス制御を組み合わせ、Apex LegendsのMOD作成用のSDKであるR5Reloadedでのダミー（ボット）に対するエイムトレーニングを支援するものです。

## 注意事項

このソフトウェアは、マウス入力の制御を目的として設計されており、ゲームのバイナリの読み書きは一切行いません。使用については、Apex LegendsのMOD作成用のSDKであるR5Reloadedでのダミー（ボット）に対するエイムトレーニング限定してください。オンラインにおける人間のプレイヤーに対する使用は倫理的に問題があり、アカウントのバンのリスクが高いため、推奨されません。

さらに、このソフトウェアは公式のApex Legendsでは使用できません。現在の実装では、敵と味方を区別することが根本的に不可能であるためです。

## 主な特徴

- YOLOv8およびTensorRT Engineを使用したリアルタイムオブジェクト検出：数十ミリ秒単位の高速な検出と高精度なターゲットロックを実現
- 非侵襲的なスクリーンキャプチャ技術：DirectXを利用し、1ms未満の超低遅延でフレームを取得
- PID制御を用いた精密なマウス制御：自然で滑らかなエイム動作を実現し、人間の動きに近い挙動を再現

## インストール手順

**注意：** このプロジェクトの推論モデルは[https://github.com/Ape-xCV/Apex-CV-YOLO-v8-Aim-Assist-Bot]との互換性を維持しています。元のプロジェクトにおいて更新されたモデルが正常に機能している場合、本プロジェクトでも使用できます。


# インストール手順

## バージョンチェックリスト:

| CUDA   | cuDNN | TensorRT | PyTorch |
| :----: | :---: | :------: | :-----: |
| 12.1   | 8.9.0 | 8.6.1.6  | 2.3.0   |

1. `r5-computer-vision.zip`を**C:\temp\r5-computer-vision**に解凍してください。

2. [`NVIDIAのウェブサイト`](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)から`CUDA 12.1.0`をインストールしてください。

3. `cuDNN 8.9.0`をインストールしてください。
   - [`NVIDIAデベロッパープログラム`](https://developer.nvidia.com/login)に登録してください。
   - [cuDNNダウンロードアーカイブ](https://developer.nvidia.com/rdp/cudnn-archive)にアクセスしてください。
   - `Download cuDNN v8.9.0 (April 11th, 2023), for CUDA 12.x`をクリックしてください。
   - `Local Installer for Windows (Zip)`をダウンロードしてください。
   - `cudnn-windows-x86_64-8.9.0.131_cuda12-archive.zip`を解凍してください。
   - 3つのフォルダ（`bin`、`include`、`lib`）をすべてコピーし、`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`に貼り付けてください。

4. [`Pythonウェブサイト`](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)から`Python 3.11.9 (64-bit)`をインストールしてください。
   - カスタムインストールから、「Add Python to environment path」にチェックを入れてください。
   - コマンドプロンプトで、以下のコマンドを順番に実行してください:
   ```shell
   cd /D C:\temp\r5-computer-vision
   python -m pip install --upgrade pip
   python -m venv r5cv
   .\r5cv\Scripts\activate
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. `TensorRT`をインストールしてください。
   - [NVIDIA TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download)にアクセスしてください。
   - [NVIDIAウェブサイト](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip)から`TensorRT 8.6 GA and CUDA 12.0 and 12.1 ZIP Package`をダウンロードしてください。
   - `TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip`を**C:\TEMP**に解凍してください。
   - TensorRTがPATHに追加されたら、そのコマンドプロンプトを閉じて新しいものを開いてください。そして以下を入力してください:
   ```shell
   cd /D C:\TEMP\TensorRT-8.6.1.6\python
   pip install tensorrt-8.6.1-cp310-none-win_amd64.whl
   ```

6. `best_8s.pt`を`best_8s.engine`にエクスポートしてください（必要な場合）:
   - **_[Win+R]_**を押して**cmd**と入力し、コマンドプロンプトを開いてください。そして以下を入力してください:
   ```shell
   set CUDA_MODULE_LOADING=LAZY
   cd /D C:\TEMP\Ape-xCV\MODEL
   yolo export model=best_8s.pt format=engine opset=12 workspace=7
   ```


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
