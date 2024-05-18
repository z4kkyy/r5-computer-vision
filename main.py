import argparse
import json
import logging
import os
import platform
import sys
import threading
from time import time

import cpuinfo
import screeninfo
import torch

import capture
import core
import listener


class LoggingFormatter(logging.Formatter):
    """
    Logging formatter that adds colors to the output.
    """
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    gray = "\x1b[38m"
    reset = "\x1b[0m"
    bold = "\x1b[1m"

    COLORS = {
        logging.DEBUG: gray + bold,
        logging.INFO: blue + bold,
        logging.WARNING: yellow + bold,
        logging.ERROR: red,
        logging.CRITICAL: red + bold,
    }

    def format(self, record) -> str:
        """
        Format the log record.

        :param record: log record
        :return: formatted string
        """
        log_color = self.COLORS[record.levelno]
        format_str = "(black){asctime}(reset) (levelcolor)[{levelname}](reset) (green)[{name}](reset) {message}"
        format_str = format_str.replace("(black)", self.black + self.bold)
        format_str = format_str.replace("(reset)", self.reset)
        format_str = format_str.replace("(levelcolor)", log_color)
        format_str = format_str.replace("(green)", self.green + self.bold)
        formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)


def generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Get the path of the current directory
    dirname = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model_dir", type=str, default=os.path.join(dirname, "model"))
    parser.add_argument("--model_name", type=str, default="best_8n.engine")  # best_8s.engine
    parser.add_argument("--verbose", type=bool, default=False)

    parser.add_argument("--wait", type=int, default=0, help="Wait time")
    parser.add_argument("--toggle_key_1", type=str, default="y", help="Toggle key to activate assist while shooting")
    parser.add_argument("--toggle_key_2", type=str, default="u", help="Toggle key to activate assist while aiming down sight")

    parser.add_argument("--Kp", type=float, default=0.35, help="Kp")  # proporcional to distance 0.4 nimble 0.1 slack
    parser.add_argument("--Ki", type=float, default=0.02, help="Ki")  # integral accumulator 0.04 explosive 0.01 composed
    parser.add_argument("--Kd", type=float, default=0.3, help="Kd")  # derivative absorber 0.4 stiff 0.1 soft

    return parser


def setup_logger(args) -> None:
    # Initialize logger
    logger = logging.getLogger("r5CV")
    if args.debug is True:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Add console and file handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LoggingFormatter())
    file_handler = logging.FileHandler(filename="r5CV.log", encoding="utf-8", mode="w")
    file_handler_formatter = logging.Formatter(
        "[{asctime}] [{levelname}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
    )
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    # Parse arguments
    parser = generate_parser()
    args = parser.parse_args()

    # Setup logger
    setup_logger(args)

    # Load config.json
    if not os.path.isfile(f"{os.path.realpath(os.path.dirname(__file__))}/config.json"):
        sys.exit()
    else:
        with open(f"{os.path.realpath(os.path.dirname(__file__))}/config.json") as file:
            config = json.load(file)

    # Get system information
    system_info = cpuinfo.get_cpu_info()
    logger = logging.getLogger("r5CV")

    print(f"""
        ██████╗ ███████╗ ██████╗██╗   ██╗
        ██╔══██╗██╔════╝██╔════╝██║   ██║    Welcome to r5 Computer Vision! 🤗🚀🎉
        ██████╔╝███████╗██║     ██║   ██║    Python version: {system_info['python_version']}
        ██╔══██╗╚════██║██║     ╚██╗ ██╔╝    Running on: {platform.platform()}
        ██║  ██║███████║╚██████╗ ╚████╔╝
        ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═══╝
    """)
    logger.info("Initializing r5CV...")
    logger.info(f"Python version: {system_info['python_version']}")
    logger.info(f"Running on: {platform.platform()}")
    logger.info(f"Detected CPU: {system_info['brand_raw']}")
    logger.info(f"Detected GPU: {torch.cuda.get_device_name()}")

    # if not run on Windows, exit
    if platform.system() != "Windows":
        logger.error("r5CV only supports Windows operating system.")
        sys.exit()

    # if main display is not 1080p, exit
    monitors = screeninfo.get_monitors()
    monitor_check = False
    for monitor in monitors:
        if monitor.width == 1920 and monitor.height == 1080:
            if monitor.is_primary:
                monitor_check = True
                break
    if not monitor_check:
        logger.error("r5CV only supports 1080p resolution.")
        sys.exit()

    # Start listening
    input_listener = listener.InputListener(args=args)
    thread = threading.Thread(target=input_listener.start_listener)
    thread.start()

    # Initialize camera
    screen_camera = capture.Capture()

    # Initialize r5CV core module
    r5CV = core.r5CVCore(
        args=args,
        config=config,
        camera=screen_camera
    )

    logger.info("Successfully initialized! Starting main process...")

    exec_count = 0
    elapsed_time = 0

    while True:
        # This is the main loop, where the inference is executed.
        # The inference process is executed in every iteration of the loop.
        # Performance:
        # - Capture: avg. 8.25 ms
        # - Inference: avg. 18.23 ms (using best_8s.engine on TensorRT)
        # - Iteration FPS: 35.97 (27.80 ms per frame)
        exec_count += 1
        time_start = time()

        key_state = input_listener.get_key_state()      # hold_state, toggle_state_1, toggle_state_2, shutdown
        mouse_state = input_listener.get_mouse_state()  # mouse_left_state, mouse_right_state, mouse_right_toggle_state

        if key_state[3]:  # shutdown = True
            logger.info("Shutting down...👋")
            sys.exit()

        r5CV.execute(key_state, mouse_state)

        elapsed_time += time() - time_start

        if exec_count % 1000 == 0:
            logger.info(f"Avg. iteration FPS: {1000 / elapsed_time:.2f} ({elapsed_time :.2f} ms per frame)")
            elapsed_time = 0
