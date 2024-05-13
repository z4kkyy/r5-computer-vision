import logging

from pynput import keyboard, mouse


class KeyListener:
    """
    Class that listens to keyboard and mouse events.
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger("r5CV")
        self.hold_active = False
        self.toggle_active = False
        self.shutdown = False

    def start_listener(self) -> None:
        """
        Start the listener.
        """
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ).start()

        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
        ).start()

    def on_press(self, key) -> None:
        """
        Handle key press events.
        if key is shift, set hold_active to True.
        if key is y, toggle toggle_active.
        if key is home, set shutdown to True.

        :param key: key pressed
        """
        prev_hold_state = self.hold_active
        if key == keyboard.Key.shift:
            self.hold_active = True
            if not prev_hold_state:
                self.logger.info("Shift hold: ON")

        if key == keyboard.KeyCode.from_char('y'):
            self.toggle_active = not self.toggle_active

        if key == keyboard.Key.home:
            self.shutdown = True
            self.logger.info("Shutting down...")

    def on_release(self, key) -> None:
        """
        Handle key release events.
        if key is shift, set hold_active to False.

        :param key: key released
        """
        if key == keyboard.Key.shift:
            self.hold_active = False
            self.logger.info("Shift hold: OFF")

    def on_click(self, x, y, button, pressed):
        pass

    def get_key_state(self) -> tuple:
        """
        Get the current state of the keys.

        :return: hold_active, toggle_active, shutdown
        """
        return self.hold_active, self.toggle_active, self.shutdown
