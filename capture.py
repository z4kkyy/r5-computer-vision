import logging
import time

import dxcam
import numpy as np
import win32api


# using dxcam (https://github.com/AI-M-BOT/DXcam)
class Capture:
    def __init__(self) -> None:
        """
        Initialize the capture object.
        The region is set to the center of the screen.
        """
        self.counter = 0
        self.screen_capture_time = 0
        self.logger = logging.getLogger("r5CV")

        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)

        # the portion of the screen to capture, 640 x 640 square in the center of the screen
        # NOTE: captured area must be a square, so crop_height == crop_width
        crop_height = 640
        crop_width = 640

        # the offset of the window
        # must be integer
        self.x_offset = (screen_width - crop_width) // 2
        self.y_offset = (screen_height - crop_height) // 2

        self.region = (
            self.x_offset,
            self.y_offset,
            self.x_offset + crop_width,
            self.y_offset + crop_height
        )
        self.camera = dxcam.create(region=self.region)

        self.camera.start(target_fps=144)

    def capture(self) -> np.ndarray:
        """
        Capture the screen. Note that dxcam.DXCamera.grab() will return None if there is no new frame since the last time called.

        :return: captured image
        """
        # NOTE: to view the image:
        # from PIL import Image
        # Image.fromarray(frame).show()
        self.counter += 1
        start_time = time.time()
        # while True:
        #     image = self.camera.grab()
        #     if image is not None:
        #         break
        image = self.camera.get_latest_frame()

        self.screen_capture_time += time.time() - start_time
        if self.counter % 1000 == 0:
            print("------------------------------------------------------------------------------")
            self.logger.info(f"Screen capture: {self.screen_capture_time:.2f} ms")
            self.screen_capture_time = 0
        return image


if __name__ == "__main__":
    print(dxcam.device_info())
    camera = Capture()
    image = camera.capture()
