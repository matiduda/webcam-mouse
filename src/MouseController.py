import pyautogui


class MouseController:
    pyautogui.FAILSAFE = False
    def __init__(self, speed, debug):
        self.speed = speed
        self.debug = debug

    def move(self, x_offset, y_offset, duration):
        if self.debug:
            print(f'Move: X: {x_offset} Y: {y_offset}')
        pyautogui.moveRel(xOffset=x_offset, yOffset=y_offset, duration=duration)

    def left_click(self):
        if self.debug:
            print("Left click!")
        pyautogui.click(button=pyautogui.PRIMARY)

    def right_click(self):
        if self.debug:
            print("Right click!")
        pyautogui.click(button=pyautogui.SECONDARY)


if __name__ == "__main__":
    # pyautogui.moveTo(100, 200, 0.5)
    pyautogui.moveRel(xOffset=0, yOffset=-100, duration=0.1)
