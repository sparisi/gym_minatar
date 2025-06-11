import imageio
import numpy as np
import gymnasium
import gym_minatar
import argparse
from pynput import keyboard
import time

# Mutable object (list) to signal when the program should exit
program_running = [True]

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("--record", action="store_true")
args = parser.parse_args()

env_id = args.env.lower()
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
if args.record:
    # Gymnasium human rendering does not return RGB array, so we must make a copy
    env_record = gymnasium.make(env_id, render_mode="rgb_array")
    frames = []

def step(action):
    if not env.action_space.contains(action):
        return
    env.step(action)
    if args.record:
        env_record.step(action)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)

def reset():
    seed = np.random.randint(999)
    env.reset(seed=seed)
    if args.record:
        env_record.reset(seed=seed)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)

def level_up():
    env.unwrapped.level_up()
    if args.record:
        env_record.unwrapped.level_up()

def level_one():
    env.unwrapped.level_one()
    if args.record:
        env_record.unwrapped.level_one()

def on_press(key):
    try:
        action = None
        if key == keyboard.Key.space:
            step(5)
        elif key == keyboard.Key.up:
            step(3)
        elif key == keyboard.Key.down:
            step(4)
        elif key == keyboard.Key.left:
            step(1)
        elif key == keyboard.Key.right:
            step(2)
        elif key == keyboard.Key.enter:
            step(0)
        elif key == keyboard.Key.shift_r:
            level_up()
        elif key == keyboard.Key.backspace:
            level_one()
        elif key == keyboard.Key.esc:
            # Can't call env.close() or pygame will freeze everything
            program_running[0] = False
            return False
        elif key.char.isalpha() and key.char == "r":
            reset()
        else:
            pass
    except AttributeError:
        pass

print(
    "\n"
    "Move: \t\t← ↑ → ↓\n"
    "Stay: \t\tENTER\n"
    "Fire: \t\tSPACEBAR\n"
    "Reset Board: \tR\n"
    "Level Up: \tRIGHT SHIFT\n"
    "Reset Level: \tBACKSPACE\n"
    "Exit: \t\tESC\n"
)

reset()

# Start listener in a non-blocking way
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while program_running[0]:
        time.sleep(0.05)
finally:
    # Cleanup in main thread
    if args.record:
        imageio.mimsave(args.env + ".gif", frames, fps=30)
    listener.stop()
    env.close()
