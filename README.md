<div id="container", align="center">
    <a href=gym_minatar/breakout.py>
        <figure>
            <img src="figures/breakout.gif" height=150 width=150 />
            <figcaption>Breakout</figcaption>
        </figure>
    </a>
    <a href=gym_minatar/space_invaders.py>
        <figure>
            <img src="figures/space_invaders.gif" height=150 width=150 />
            <figcaption>Space Invaders</figcaption>
        </figure>
    </a>
    <a href=gym_minatar/freeway.py>
        <figure>
            <img src="figures/freeway.gif" height=150 width=150 />
            <figcaption>Freeway</figcaption>
        </figure>
    </a>
    <a href=gym_minatar/asterix.py>
        <figure>
            <img src="figures/asterix.gif" height=150 width=150 />
            <figcaption>Asterix</figcaption>
        </figure>
    </a>
    <a href=gym_minatar/seaquest.py>
        <figure>
            <img src="figures/seaquest.gif" height=150 width=150 />
            <figcaption>Seaquest</figcaption>
        </figure>
    </a>
</div>

## Overview
Collection of simplified [Atari](https://gymnasium.farama.org/environments/atari/)
games fully compatible with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
Inspired by [MinAtar](https://github.com/kenjyoung/MinAtar).

#### Gym-MinAtar vs MinAtar
- All games are rendered with [PyGame](https://www.pygame.org/news) rather than
  [Matplotlib](https://matplotlib.org/), as in classic
  [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments.
- Different observation spaces. In MinAtar, the observation space has a separate
  channel for every entity in the game with binary values (0/1). Gym-MinAtar
  has a lower dimensional observation space with ternary values (-1/0/1).
  For example, in MinAtar's Space Invaders, aliens moving left and aliens moving
  right are encoded in two separate channels. Instead, Gym-MinAtar uses one
  channel with -1 for aliens moving left, and 1 for aliens moving right.
- Different rendering scheme. MinAtar uses one pixel for trails (like
  car trails), with different shades for different speeds. Gym-MinAtar uses the
  same shades, but trails are longer for faster cars.
- Game-specific dynamics are different (like cooldown times and speeds).

### Install and Make an Environment
```
pip install -e .
```

```python
import gymnasium
import gym_minatar
env = gymnasium.make("Gym-MinAtar/SpaceInvaders-v1", render_mode="human")
env.reset()
env.step(1) # LEFT
env.step(3) # SHOOT
env.render()
```

### Playground
```
pip install -e .[playground]
python playground.py breakout
```
This will start a Breakout game (commands are displayed on the terminal).
The flag `--record` allows you to record the game and save it to a GIF.
The flag `--practice` makes the game wait until press a key to act.

## Games
### `Gym-MinAtar/Breakout-v1`
<div id="container">
  <a href=gym_minatar/breakout.py>
        <figure>
            <img src="figures/breakout.gif" height=150 width=150 />
        </figure>
    </a>
</div>

### `Gym-MinAtar/SpaceInvaders-v1`
<div id="container">
  <a href=gym_minatar/space_invaders.py>
        <figure>
            <img src="figures/space_invaders.gif" height=150 width=150 />
        </figure>
    </a>
</div>

### `Gym-MinAtar/Freeway-v1`
<div id="container">
    <a href=gym_minatar/freeway.py>
        <figure>
            <img src="figures/freeway.gif" height=150 width=150 />
        </figure>
    </a>
</div>

### `Gym-MinAtar/Asterix-v1`
<div id="container">
    <a href=gym_minatar/asterix.py>
        <figure>
            <img src="figures/asterix.gif" height=150 width=150 />
        </figure>
    </a>
</div>

### `Gym-MinAtar/Seaquest-v1`
<div id="container">
    <a href=gym_minatar/seaquest.py>
        <figure>
            <img src="figures/seaquest.gif" height=150 width=150 />
        </figure>
    </a>
</div>






## Default MDP (`Gridworld` Class)

### <ins>Action Space</ins>
The action is discrete in the range `{0, 4}` for `{LEFT, DOWN, RIGHT, UP, STAY}`.

### <ins>Observation Space</ins>
&#10148; <strong>Default</strong>  
The observation is discrete in the range `{0, n_rows * n_cols - 1}`.
Each integer denotes the current location of the agent.
For example, in a 3x3 grid the states are
```
 0 1 2
 3 4 5
 6 7 8
```

&#10148; <strong>Board</strong>  
If you prefer to observe the `(row, col)` index of the current position of the
agent, make the environment with the `coordinate_observation=True` argument.

&#10148; <strong>RGB</strong>  
To use classic RGB pixel observations, make the environment with
`render_mode="rgb_array"`.

### <ins>Starting State</ins>
The episode starts with the agent at the top-left tile. Make new classes for
different starting states. For example, in `GridworldMiddleStart` the agent starts
in the middle of the grid, while in `GridworldRandomStart` it starts in a random tile.

### <ins>Transition</ins>
By default, the transition is deterministic except in quicksand tiles,
where any action fails with 90% probability (the agent does not move).  
Transition can be made stochastic everywhere by passing `random_action_prob`.
This is the probability that the action will be random.
For example, if `random_action_prob=0.1` there is a 10% chance that the agent
will do a random action instead of doing the one passed to `self.step(action)`.  

### <ins>Rewards</ins>
- Doing `STAY` at the goal: +1
- Doing `STAY` at a distracting goal: 0.1
- Any action in penalty tiles: -10
- Any action in small penalty tiles: -0.1
- Walking on a pit tile: -100
- Otherwise: 0

&#10148; <strong>Noisy Rewards</strong>  
White noise can be added to all rewards by passing `reward_noise_std`,
or only to nonzero rewards with `nonzero_reward_noise_std`.

&#10148; <strong>Auxiliary Rewards</strong>  
An auxiliary negative reward based on the Manhattan distance to the closest
goal can be added by passing `distance_reward=True`. The distance is scaled
according to the size of the grid.

### <ins>Episode End</ins>
By default, an episode ends if any of the following happens:
- A positive reward is collected (termination),
- Walking on a pit tile (termination),
- The length of the episode is `max_episode_steps` (truncation).
