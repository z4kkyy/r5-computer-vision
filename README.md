# Release Notes: r5-computer-vision

This repository is a refactored & updated version of the original Ape-xCV project.

## Special Thanks
We'd like to extend our gratitude to:
- Ape-xCV, whose project forms the base of our current work. [https://github.com/Ape-xCV/Apex-CV-YOLO-v8-Aim-Assist-Bot]
- Franklin-Zhang0, whose original efforts laid the groundwork for Ape-xCV's project. [https://github.com/Franklin-Zhang0/Yolo-v8-Apex-Aim-assist]

## Installation Instructions
To get started, please refer to the README.md in Ape-xCV’s project. The installation process is basically the same.

**Note:** Models in this project maintains compatibility with the original ones. If updated models from the original project are functioning properly, they can be utilized here too.

## Core Functionality
**Operational Workflow**: this program loops the following sequence by executing the `main.py` script:
  1. Start thread to asynchronously updates key states. (main.py, line 81)
  2. In each main loop iteration, reads the key & mouse states (main.py, line 111) and execute r5cVCore's execute method (main.py, line 118).
    The execute method:
      1. Captures the image.
      2. Do object detection.
      3. (optional) Display the image with bounding box.
      4. If conditions are met, move the mouse cursor.

**Usage**
- Shift
    - Hold shift key to **lock on target**.
- Toggle 1
    - Toggled by pressing `'y'`. You can change the key in config.yaml.
    - Use to **lock on target**, and **not while firing**.
- Toggle 2 (You need to change your ADS from **toggle** to **hold**.)
    - Toggle by pressing `'u'`. You can change the key in config.yaml.
    - Use to **lock on target while scoping**.
- HOME
    - Press home key to terminate the whole process.

## Configuration Updates
- **Parameter Management**: Transitioned most command-line arguments to `config.yaml` for streamlined configuration, though `--debug` and `--verbose` remain as command-line options. (Transition is not yet complete)

## System Enhancements
- **Logging**: Implemented a new logging system, with configurations located in `logger_config.py`.
- **Structural Improvements**: Enhanced code readability and maintainability by removing global variables and introducing an object-oriented approach. Core functionalities, such as inference and mouse control, are now encapsulated within the `r5cvCore` class in `core.py`.


## Technical Adjustments
- **Improved Capture Method**: In `capture.py`, initiated capture mode with `self.camera.start()` on line 41. This method allows the use of `.get_latest_frame` to retrieve the most recent frame from the frame buffer, which is more efficient than the previous `.grab` method.

- **PID Parameter Tuning**: Updated the default values of the PID parameters to Kp: 0.2, Ki: 0.01
, Kd: 0.08 in order to archive natural aim assist. The previous values caused excessively rapid mouse movements, making control difficult. These can be adjusted in `config.yaml`.

## Upcoming Updates

Stay tuned for more enhancements in the near future!







Here's an updated version of the release notes with the suggested changes applied for better clarity and readability:

---

# Release Notes: r5-computer-vision

This repository is a refactored & updated version of the original Ape-xCV project.

## **Special Thanks**
We'd like to extend our gratitude to:
1. **Ape-xCV**, whose project forms the base of our current work.
2. **Franklin-Zhang0**, whose original efforts laid the groundwork for Ape-xCV's project.

## **Installation Instructions**
To get started, please refer to the **README.md** in Ape-xCV’s project. The installation process remains fundamentally the same.

**Note:** Models in this project maintain compatibility with the original ones. If updated models from the original project are functioning properly, they can be utilized here too.

## **Core Functionality**
**Operational Workflow**: this program loops the following sequence by executing the `main.py` script:
  - Start thread to asynchronously update key states. (`main.py`, line 81)
  - In each main loop iteration:
    - Reads the key & mouse states (`main.py`, line 111)
    - Executes r5cVCore's execute method (`main.py`, line 118), which includes:
      - Getting latest frame(screenshot) from DXCamera instance in less than 1ms
      - Performing object detection.
      - Optionally displaying the image with bounding box.
      - If conditions are met, moving the mouse cursor.

## **Configuration Updates**
- **Parameter Management**: Transitioned most command-line arguments to `config.yaml` for streamlined configuration. Command-line options such as `--debug` and `--verbose` remain. (Transition is not yet complete)

## **System Enhancements**
- **Logging**: Implemented a new logging system, with configurations located in `logger_config.py`.
- **Structural Improvements**: Enhanced code readability and maintainability by removing global variables and introducing an object-oriented approach. Core functionalities, such as inference and mouse control, are now encapsulated within the `r5cvCore` class in `core.py`.

## **Technical Adjustments**
- **Improved Capture Method**: In `capture.py`, initiated capture mode with `self.camera.start()` on line 41. This method allows for the use of `.get_latest_frame` to retrieve the most recent frame from the frame buffer, which is more efficient than the previous `.grab` method.
- **PID Parameter Tuning**: Updated the default values of the PID parameters to Kp: 0.20, Ki: 0.01, Kd: 0.08 in order to achieve more natural aim assist. The previous values caused excessively rapid mouse movements, making control difficult. These can be adjusted in `config.yaml`.

## **Upcoming Updates**
Stay tuned for more enhancements in the near future!

---

どうでしょうか？さらに変更を加えたい点があれば教えてください。また、このリリースノートで視覚的な要素や図を追加することをお勧めします。それにより、ユーザーが変更点やワークフローをより明確に理解できるようになるでしょう。
