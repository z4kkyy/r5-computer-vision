import logging

from pynput import keyboard, mouse


class InputListener:
    def __init__(self, args) -> None:
        self.logger = logging.getLogger("r5CV")
        self.args = args
        # Initialize key states
        self.hold_active = False
        self.toggle_active = False
        self.shutdown = False
        # Initialize mouse states
        self.mouse_left_active = False
        self.mouse_right_active = False

    def start_listener(self) -> None:
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        ).start()

        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
        ).start()

    def on_key_press(self, key) -> None:
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
                self.logger.debug("Shift hold: ON")

        if key == keyboard.KeyCode.from_char(self.args.toggle_key):
            self.toggle_active = not self.toggle_active

        if key == keyboard.Key.home:
            self.shutdown = True

    def on_key_release(self, key) -> None:
        """
        Handle key release events.
        if key is shift, set hold_active to False.

        :param key: key released
        """
        if key == keyboard.Key.shift:
            self.hold_active = False
            self.logger.debug("Shift hold: OFF")

    def on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            if pressed:
                self.mouse_left_active = True
            else:
                self.mouse_left_active = False

        if button == mouse.Button.right:
            if pressed:
                self.mouse_right_active = True
            else:
                self.mouse_right_active = False

    def get_key_state(self) -> tuple:
        """
        Get the current state of the keys.

        :return: hold_active, toggle_active, shutdown
        """
        return self.hold_active, self.toggle_active, self.shutdown

    def get_mouse_state(self) -> tuple:
        """
        Get the current state of the mouse buttons.

        :return: mouse_left_active, mouse_right_active
        """
        return self.mouse_left_active, self.mouse_right_active
