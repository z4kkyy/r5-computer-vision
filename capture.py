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
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)

        # the portion of the screen to capture, 1/2 for 1080p
        # must be integer
        crop_height = 540    # does not work if 640
        crop_width = 540 * screen_width // screen_height

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

    def capture(self) -> np.ndarray:
        """
        Capture the screen. Note that dxcam.DXCamera.grab() will return None if there is no new frame since the last time called.

        NOTE: To view the image:
        from PIL import Image
        Image.fromarray(frame).show()

        :return: captured image
        """
        while True:
            image = self.camera.grab()
            if image is not None:
                return image


if __name__ == "__main__":
    print(dxcam.device_info())
    camera = Capture()
    image = camera.capture()
