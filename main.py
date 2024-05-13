import argparse
import logging
import os
import platform
import sys
import threading
from time import sleep

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
        format = "(black){asctime}(reset) (levelcolor)[{levelname}](reset) (green)[{name}](reset) {message}"
        format = format.replace("(black)", self.black + self.bold)
        format = format.replace("(reset)", self.reset)
        format = format.replace("(levelcolor)", log_color)
        format = format.replace("(green)", self.green + self.bold)
        formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)


def generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Get the path of the current directory
    dirname = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument("--model_dir", type=str, default=os.path.join(dirname, "model"))
    parser.add_argument("--model_name", type=str, default="best_8s.engine")
    parser.add_argument("--verbose", type=bool, default=False)

    parser.add_argument("--wait", type=int, default=0, help="wait time")
    parser.add_argument("--toggle_key", type=str, default="y", help="toggle key")

    parser.add_argument("--Kp", type=float, default=0.35, help="Kp")  # proporcional to distance 0.4 nimble 0.1 slack
    parser.add_argument("--Ki", type=float, default=0.02, help="Ki")  # integral accumulator 0.04 explosive 0.01 composed
    parser.add_argument("--Kd", type=float, default=0.3, help="Kd")  # derivative absorber 0.4 stiff 0.1 soft

    return parser


if __name__ == "__main__":

    logger = logging.getLogger("r5CV")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LoggingFormatter())
    # File handler
    file_handler = logging.FileHandler(filename="r5CV.log", encoding="utf-8", mode="w")
    file_handler_formatter = logging.Formatter(
        "[{asctime}] [{levelname:^8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
    )
    file_handler.setFormatter(file_handler_formatter)

    # Add the handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Welcome to r5CV!")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Running on: {platform.platform()}, {platform.machine()}")
    # logger.info(f"Processor: {platform.processor()}")

    # Parse arguments
    parser = generate_parser()
    args = parser.parse_args()

    # Start listening
    key_listener = listener.KeyListener()
    thread = threading.Thread(target=key_listener.start_listener)
    thread.start()

    active = True

    logger.info("Listening now starting...")

    # test
    while active:
        hold_active, toggle_active, shutdown = key_listener.get_key_state()

        if shutdown:
            active = False
            sys.exit()

        sleep(1)
