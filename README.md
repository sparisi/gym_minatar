<div align="center">
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
Actions are discrete, while observations have shape `(rows, cols, channels)`
with ternary values (-1, 0, or 1).
The number of actions and channels depends on the game. Below are some details
about the games. For full details, please see docs in the source code.  
All boards have size (10, 10) by default. To change it:
```python
gymnasium.make(..., size=(rows, cols))
```
To train from RGB images:
```python
import gymnasium
import gym_minatar
env = gymnasium.make("Gym-MinAtar/SpaceInvaders-v1", render_mode="rgb_array", window_size=(84, 84))
env = gymnasium.wrappers.AddRenderObservation(env, render_only=True)
```

### [`Gym-MinAtar/Breakout-v1`](gym_minatar/breakout.py)
<div>
    <figure>
        <img src="figures/breakout.gif" height=150 width=150 />
    </figure>
</div>

### [`Gym-MinAtar/SpaceInvaders-v1`](gym_minatar/space_invaders.py)
<div>
    <figure>
        <img src="figures/space_invaders.gif" height=150 width=150 />
    </figure>
</div>

### [`Gym-MinAtar/Freeway-v1`](gym_minatar/freeway.py)
<div>
    <figure>
        <img src="figures/freeway.gif" height=150 width=150 />
    </figure>
</div>

### [`Gym-MinAtar/Asterix-v1`](gym_minatar/asterix.py)
<div>
    <figure>
        <img src="figures/asterix.gif" height=150 width=150 align="left"/>
    </figure>
    The player (green) has to collect treasures (blue) while avoiding enemies (red).
    Entities (treasures and enemies) move at different speed, denoted by the trail behind them
    (longer trails means faster entity).  
    After an entity leaves the screen (or is collected, if treasure) some time must
    pass before a new one randomly appears in the same row.
    Over time, entity speed increases and respawn wait time decreases.  
    The game ends when the player is collides with an enemy.  
    The player has 5 actions (`LEFT, DOWN, RIGHT, UP, NO-OP`) and the observation space
    has 3 channels for (in order): player position (1), enemies position and
    trail (-1 moving left, 1 moving right), treasures position and trail (-1 moving
    left, 1 moving right).
</div>

### [`Gym-MinAtar/Seaquest-v1`](gym_minatar/seaquest.py)
<div>
    <figure>
        <img src="figures/seaquest.gif" height=150 width=150 />
    </figure>
</div>
