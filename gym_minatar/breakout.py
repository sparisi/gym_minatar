import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2

BLUE = (0, 0, 255)  # ball
CYAN = (0, 255, 255)  # ball trail
GREEN = (0, 255, 0)  # player
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)  # bricks


class Breakout(gym.Env):
    """
    The player controls a paddle to bounce a ball and break bricks.
    - The paddle moves left/right at the bottom of the grid, or may not move at all.
    - The ball moves diagonally and bounces off walls, paddle, and bricks.
    - The player receives a reward for breaking bricks.
    - When all bricks are destroyed, a new game starts at increased difficulty.
      - Difficulty increases ball speed.
    - The game ends if the ball falls below the paddle.
    - The observation space is a 3-channel grid with 0s for empty tiles, and
      values in [-1, 1] for moving entities:
        - Channel 0: bricks (1s).
        - Channel 1: paddle position (1).
        - Channel 2: ball and its trail (-1 moving up, 1 moving down).
        - Intermediate values in (-1, 1) denote the ball speed when it moves slower
          than 1 tile per step.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (10, 10),
        brick_rows: int = 3,
        levels: int = 3,
        window_size: tuple = None,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        self.brick_rows = brick_rows

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        assert (  # One empty row before and after the bricks, one row for the paddle
            self.brick_rows + 3 < self.n_rows
        ), f"cannot fit {brick_rows} brick rows in a board with {self.n_rows} rows"

        # First channel for paddle position.
        # Second channel for bricks.
        # Third channel for ball and its trail (-1 moving up, 1 moving down).
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3),
        )
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
        }

        self.bricks = np.zeros((self.n_rows, self.n_cols), dtype=np.int64)
        self.paddle_pos = None
        self.ball_pos = None
        self.last_ball_pos = []
        self.ball_dir = None
        self.last_action = None
        self.contact_pos = None  # Used to store ball contact with brick or paddle

        # These two variable control the ball speed: when ball_timesteps == ball_delay,
        # the ball moves. A delay of 1 means that the paddle moves x2 times
        # faster than the ball.
        # When difficulty increases, ball_delay decreases and can become negative.
        # Negative delay means that the ball is faster than the paddle.
        self.ball_delay_levels = np.arange(levels - 1, -1, -1) - levels // 2
        self.level = 0
        self.ball_delay = self.ball_delay_levels[self.level]
        self.ball_timesteps = self.ball_delay

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        if window_size is not None:
            assert np.all(np.array(window_size) >= np.array(size)), f"window size too small {window_size} for the board size {size}"
            self.window_size = window_size
        else:
            self.window_size = (
                min(64 * self.n_cols, 512),
                min(64 * self.n_rows, 512),
            )  # fmt: skip
        self.tile_size = (
            self.window_size[0] // self.n_cols,
            self.window_size[1] // self.n_rows,
        )  # fmt: skip

    def get_state(self):
        state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        state[..., 1] = self.bricks
        state[self.paddle_pos[0], self.paddle_pos[1], 0] = 1
        state[self.ball_pos[0], self.ball_pos[1], 2] = self.ball_dir[0]
        if self.contact_pos is not None:
            state[self.contact_pos[0], self.contact_pos[1], 2] = self.ball_dir[0]
        for ball_pos in self.last_ball_pos:
            if self.ball_timesteps < self.ball_delay:
                speed_scaling = (self.ball_timesteps + 0.5) / self.ball_delay
            else:
                speed_scaling = 1.0
            state[ball_pos[0], ball_pos[1], 2] = self.ball_dir[0] * speed_scaling
        return state

    def level_one(self):
        self.level = 0
        self.ball_delay = self.ball_delay_levels[self.level]

    def level_up(self):
        self.level = min(self.level + 1, len(self.ball_delay_levels) - 1)
        self.ball_delay = self.ball_delay_levels[self.level]

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.paddle_pos = [
            self.n_rows - 1,
            self.np_random.integers((self.n_cols)),
        ]
        self.ball_pos = [
            self.np_random.integers(self.brick_rows + 1, self.n_rows - 1),
            self.np_random.integers(self.n_cols),
        ]  # Do not spawn in paddle or brick rows
        self.ball_dir = [
            -1,
            self.np_random.choice((-1, 1)),
        ]  # Always spawn going up
        self.ball_delay = self.ball_delay_levels[self.level]
        self.ball_timesteps = self.ball_delay
        self.bricks[:] = 0
        self.bricks[1 : self.brick_rows + 1, :] = 1
        self.last_action = None
        self.last_ball_pos = []
        self.contact_pos = None

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _step(self, action: int):
        self.contact_pos = None
        terminated = False
        reward = 0.0

        # Move paddle
        if action == NOP:
            pass
        elif action == LEFT:
            self.paddle_pos[1] = max(self.paddle_pos[1] - 1, 0)
        elif action == RIGHT:
            self.paddle_pos[1] = min(self.paddle_pos[1] + 1, self.n_cols - 1)
        else:
            raise ValueError("illegal action")

        # Check if it's time to move the ball
        if self.ball_delay > 0:
            if self.ball_timesteps < self.ball_delay:
                self.ball_timesteps += 1
                return self.get_state(), reward, terminated, False, {}
            else:
                self.ball_timesteps = 0
                ball_steps = 1
        else:
            ball_steps = abs(self.ball_delay) + 1

        def where_ball_is_going(position, direction):
            front_pos = [  # Vertical
                position[0] + direction[0],
                position[1],
            ]
            diag_pos = [
                position[0] + direction[0],
                position[1] + direction[1],
            ]
            side_pos = [  # Horizontal
                position[0],
                position[1] + direction[1],
            ]
            return front_pos, diag_pos, side_pos

        self.last_ball_pos = []
        for steps in range(ball_steps):
            self.last_ball_pos.append(self.ball_pos)
            new_ball_pos = [
                self.ball_pos[0] + self.ball_dir[0],
                self.ball_pos[1] + self.ball_dir[1],
            ]

            # Collision with bricks can't happen when collision with ceiling or floor happens
            check_for_bricks = True

            # Collision with side walls
            if new_ball_pos[1] < 0 or new_ball_pos[1] >= self.n_cols:
                new_ball_pos[1] = min(max(new_ball_pos[1], 0), self.n_cols - 1)
                self.ball_dir[1] *= -1  # Bounce

            # Collision with ceiling
            if new_ball_pos[0] < 0:
                new_ball_pos[0] = 0
                self.ball_dir[0] *= -1
                check_for_bricks = False

            # Collision with floor or paddle
            elif new_ball_pos[0] == self.n_rows - 1:
                check_for_bricks = False
                game_over = True
                front_pos, diag_pos, side_pos = where_ball_is_going(self.ball_pos, self.ball_dir)
                if front_pos == self.paddle_pos:  # Keep side direction and bounce up
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    game_over = False
                    self.contact_pos = front_pos
                elif diag_pos == self.paddle_pos:  # Bounce back (diagonally)
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.ball_dir[1] *= -1
                    game_over = False
                    self.contact_pos = diag_pos
                elif side_pos == self.paddle_pos:  # Keep down direction and side bounce
                    new_ball_pos = self.ball_pos
                    self.ball_dir[1] *= -1
                    game_over = True  # Hitting the paddle from the side does not save the ball
                    self.contact_pos = side_pos
                if game_over:
                    terminated = True
                    self.level_one()
                    self.reset()
                    return self.get_state(), reward, terminated, False, {}

            # Collision with brick (must check after wall collision)
            if check_for_bricks:
                front_pos, diag_pos, side_pos = where_ball_is_going(self.ball_pos, self.ball_dir)
                if self.bricks[front_pos[0], front_pos[1]]:
                    reward = 1.0
                    self.bricks[front_pos[0], front_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.contact_pos = front_pos
                elif self.bricks[diag_pos[0], diag_pos[1]]:
                    reward = 1.0
                    self.bricks[diag_pos[0], diag_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.ball_dir[1] *= -1
                    self.contact_pos = diag_pos
                elif self.bricks[side_pos[0], side_pos[1]]:
                    reward = 1.0
                    self.bricks[side_pos[0], side_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[1] *= -1
                    self.contact_pos = side_pos

            self.ball_pos = new_ball_pos

            if self.bricks.sum() == 0:
                if self.level == len(self.ball_delay_levels) - 1:
                    terminated = True
                else:
                    self.level_up()
                    self.reset()

        self.last_action = action
        return self.get_state(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption(self.unwrapped.spec.id)
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        state = self.get_state()

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        def draw_tile(row, col, color, scale=1.0):
            pos = (col * self.tile_size[0], row * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            if scale != 1.0:
                rect = rect.scale_by(scale)
            pygame.draw.rect(self.window_surface, color, rect)

        # Draw bricks
        for x in range(1, self.brick_rows + 1):
            for y in range(self.n_cols):
                if self.bricks[x, y]:
                    draw_tile(x, y, GRAY)

        # Draw trail
        for trail in self.last_ball_pos:
            if self.ball_timesteps < self.ball_delay:
                speed_scaling = (self.ball_timesteps + 0.5) / self.ball_delay
            else:
                speed_scaling = 1.0
            draw_tile(trail[0], trail[1], CYAN, speed_scaling)

        # Draw ball
        draw_tile(self.ball_pos[0], self.ball_pos[1], BLUE)

        # Draw trail if there was a collision
        if self.contact_pos is not None:
            draw_tile(self.contact_pos[0], self.contact_pos[1], CYAN)

        # Draw paddle
        draw_tile(self.paddle_pos[0], self.paddle_pos[1], GREEN)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
        else:
            raise NotImplementedError

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
