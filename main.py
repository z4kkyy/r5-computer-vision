import argparse
import logging
import logging.config
import os
import platform
import sys
import threading
from time import time

import cpuinfo
import torch
import yaml

import capture
import core
import listener
from logger_config import setup_logger

# NOTE: PID controller parameters
# Kp: proporcional to distance 0.4 nimble 0.1 slack
# Ki: integral accumulator 0.04 explosive 0.01 composed
# Kd: derivative absorber 0.4 stiff 0.1 soft


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")

    args = parser.parse_args()

    # get system information
    system_info = cpuinfo.get_cpu_info()
    logger = logging.getLogger("r5CV")

    # setup logger
    setup_logger(args.debug)

    # print welcome message
    version_info = sys.version_info
    formatted_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}.{version_info.releaselevel}.{version_info.serial}"
    bitness = "64bit" if sys.maxsize > 2**32 else "32bit"

    print(f"""
        ██████╗ ███████╗ ██████╗██╗   ██╗
        ██╔══██╗██╔════╝██╔════╝██║   ██║    Welcome to r5 Computer Vision! 🤗🚀🎉
        ██████╔╝███████╗██║     ██║   ██║    Python version: {formatted_version} ({bitness})
        ██╔══██╗╚════██║██║     ╚██╗ ██╔╝
        ██║  ██║███████║╚██████╗ ╚████╔╝
        ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═══╝
    """)
    logger.info("Initializing r5CV...")
    logger.info(f"Python version: {formatted_version} ({bitness})")
    logger.info(f"Detected CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    logger.info(f"Detected GPU: {torch.cuda.get_device_name()}")

    # if not run on Windows, exit
    if platform.system() != "Windows":
        logger.error("r5CV only supports Windows operating system.")
        sys.exit()

    # # if main display is not 1080p, exit
    # monitors = screeninfo.get_monitors()
    # monitor_check = False
    # for monitor in monitors:
    #     if monitor.width == 1920 and monitor.height == 1080:
    #         if monitor.is_primary:
    #             monitor_check = True
    #             break
    # if not monitor_check:
    #     logger.error("r5CV only supports 1080p resolution.")
    #     sys.exit()

    # load config
    with open(f"{os.path.realpath(os.path.dirname(__file__))}/config.yaml", "rt") as file:
        config = yaml.safe_load(file.read())

    # start listening
    input_listener = listener.InputListener(config=config)
    thread = threading.Thread(target=input_listener.start_listener)
    thread.start()

    # initialize camera
    screen_camera = capture.Capture()

    # initialize r5CV core module
    r5CV = core.r5CVCore(
        args=args,
        config=config,
        camera=screen_camera
    )

    logger.info("Successfully initialized! Starting main process...")

    exec_count = 0
    elapsed_time = 0
    readstate_time = 0

    while True:
        # This is the main loop, where the inference is executed.
        # The inference process is executed in every iteration of the loop.
        # Performance:
        # - Capture: avg. 8.25 ms
        # - Inference: avg. 18.23 ms (using best_8s.engine on TensorRT)
        # - Iteration FPS: 51.97 (27.80 ms per frame)
        exec_count += 1

        # read key and mouse states
        key_state = input_listener.get_key_state()      # hold_state, toggle_state_1, toggle_state_2, shutdown
        mouse_state = input_listener.get_mouse_state()  # mouse_left_state, mouse_right_state
        if key_state[3]:  # shutdown = True
            logger.info("Shutting down...👋")
            sys.exit()
        time_start = time()
        r5CV.execute(key_state, mouse_state)
        elapsed_time += time() - time_start

        if exec_count % 1000 == 0:
            logger.info(f"Avg. iteration FPS: {1000 / elapsed_time:.2f} ({elapsed_time :.2f} ms per frame)")

            elapsed_time = 0
