import logging
import sys


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


def setup_logger(debug) -> None:
    # Initialize logger
    logger = logging.getLogger("r5CV")
    if debug is True:
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
