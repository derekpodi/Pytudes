#! python3
# noSleep.py - small infrequent mouse nudge - prevents idle on computer screen

import pyautogui
import random

def north():
    print("Move North")
    return pyautogui.moveRel(0, -10, 0.25)

def south():
    print("Move South")
    return pyautogui.moveRel(0, 10, 0.25)

def east():
    print("Move East")
    return pyautogui.moveRel(10, 0, 0.25)

def west():
    print("Move West")
    return pyautogui.moveRel(-10, 0, 0.25)


def switch(direction: int):
    def invalid(): 
        return "Invalid direction"

    switch = {
        1: north,
        2: south,
        3: east,
        4: west,
    }
    return switch.get(direction, invalid)()


def noSleep():
    """
    Slightly move mouse (few pixels) every few seconds to not go idle
    Input:
        None
    Return:
        None
    """
    print("Press CTRL-C to quit.")

    try:
        while True:
            direction = random.randint(1,4)

            switch(direction)

            print("Movement starting in ", end="")
            pyautogui.countdown(10)
            # time.sleep(10)

    except KeyboardInterrupt:
        print("Process has stopped...")


if __name__ == "__main__":
    noSleep()