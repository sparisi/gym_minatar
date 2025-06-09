# This script is designed to work on Windows, macOS, and Linux.
# It detects physical key presses regardless of keyboard language layout.

import gymnasium
import gym_minatar
import argparse
from pynput import keyboard
import time

# A flag to signal when the program should exit.
# We use a mutable object (a list) to ensure changes are seen across threads.
program_running = [True]

parser = argparse.ArgumentParser()
parser.add_argument("env")

env_id = parser.parse_args().env.lower()
if "seaquest" in env_id:
    env_id = "Gym-MinAtar/Seaquest-v1"
elif "breakout" in env_id:
    env_id = "Gym-MinAtar/Breakout-v1"
elif "asterix" in env_id:
    env_id = "Gym-MinAtar/Asterix-v1"
elif "freeway" in env_id:
    env_id = "Gym-MinAtar/Freeway-v1"
elif "space_invaders" in env_id:
    env_id = "Gym-MinAtar/SpaceInvaders-v1"
else:
    raise ValueError("game not found")

env = gymnasium.make(env_id, render_mode="human")
env.reset()

def on_press(key):
    try:
        if key == keyboard.Key.space:
            env.step(5)
        elif key == keyboard.Key.up:
            env.step(1)
        elif key == keyboard.Key.down:
            env.step(2)
        elif key == keyboard.Key.left:
            env.step(3)
        elif key == keyboard.Key.right:
            env.step(4)
        elif key == keyboard.Key.enter:
            env.reset()
        elif key == keyboard.Key.esc:
            # env.close()
            program_running[0] = False
            return False
        else:
            print(f"invalid action")
    except AttributeError:
        print(f"invalid action")

# Start listener in a non-blocking way
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Main loop
try:
    while program_running[0]:
        time.sleep(0.05)
finally:
    # Cleanup in main thread
    listener.stop()
    env.close()
