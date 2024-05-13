import win32api
import dxcam
import numpy as np


# using dxcam (https://github.com/AI-M-BOT/DXcam)
class Capture:
    def __init__(self) -> None:
        """
        Initialize the capture object.
        The region is set to the center of the screen.
        """
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)

        # the portion of the screen to capture, 1/2 for 1080p
        # must be integer
        crop_height = 540
        crop_width = 540 * screen_width // screen_height

        # the offset of the window
        # must be integer
        x_offset = (screen_width - crop_width) // 2
        y_offset = (screen_height - crop_height) // 2

        self.region = (
            x_offset,
            y_offset,
            x_offset + crop_width,
            y_offset + crop_height
        )
        self.camera = dxcam.create(region=self.region)

    def capture(self) -> np.ndarray:
        """
        Capture the screen.
        """
        while True:
            image = self.camera.grab()
            if image is not None:
                return image


if __name__ == "__main__":
    print(dxcam.device_info())
    camera = Capture()
    image = camera.capture()
