import logging

from pynput import keyboard, mouse


class InputListener:
    def __init__(self, config) -> None:
        self.logger = logging.getLogger("r5CV")
        self.config = config

        # Initialize key states
        self.hold_state = False
        self.toggle_state_1 = False  # lock on target
        self.toggle_state_2 = False  # lock on target when ADS
        self.shutdown = False

        # Initialize mouse states
        self.mouse_left_state = False
        self.mouse_right_state = False

    def start_listener(self) -> None:
        """
        Start the input listener.
        """
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        ).start()

        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
        ).start()

    def on_key_press(self, key) -> None:
        prev_hold_state = self.hold_state
        if key == keyboard.Key.shift:
            self.hold_state = True
            if not prev_hold_state:
                self.logger.debug("Holding shift starts...")

        if key == keyboard.KeyCode.from_char(self.config["toggle_key_1"]):
            self.toggle_state_1 = not self.toggle_state_1
            state = "ON" if self.toggle_state_1 else "OFF"
            self.logger.debug(f"Toggle 1 state: {state}")

        if key == keyboard.KeyCode.from_char(self.config["toggle_key_2"]):
            self.toggle_state_2 = not self.toggle_state_2
            state = "ON" if self.toggle_state_2 else "OFF"
            self.logger.debug(f"Toggle 2 state: {state}")

        if key == keyboard.Key.home:
            self.shutdown = True

    def on_key_release(self, key) -> None:
        if key == keyboard.Key.shift:
            self.hold_state = False
            self.logger.debug("Holding shift ends...")

    def on_mouse_click(self, x, y, button, pressed) -> None:
        if button == mouse.Button.left:
            if pressed:
                self.mouse_left_state = True
            else:
                self.mouse_left_state = False

        if button == mouse.Button.right:
            if pressed:
                self.mouse_right_state = True
            else:
                self.mouse_right_state = False

    def get_key_state(self) -> tuple:
        return self.hold_state, self.toggle_state_1, self.toggle_state_2, self.shutdown

    def get_mouse_state(self) -> tuple:
        return self.mouse_left_state, self.mouse_right_state
