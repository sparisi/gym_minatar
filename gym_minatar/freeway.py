import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
UP = 1
DOWN = 2

RED = (255, 0, 0)  # cars
PALE_RED = (255, 155, 155)  # car trails
GREEN = (0, 255, 0)  # player
BLACK = (0, 0, 0)


class Freeway(gym.Env):
    """
    The player (chicken) starts at the bottom of the screen and must reach the top,
    crossing lanes of moving cars.
    - The player can move up/down or may not move at all.
    - Cars move horizontally at different (random) speeds and directions,
      wrapping around the screen.
    - The game ends when the player is hit by a car.
    - Each time the player reaches the top, a reward of +1 is given and the car
      speeds increase (still random, though).
    - The observation space is a 3-channel grid with 0s for empty tiles, and 1 or -1
      for information about the game entities:
        - Channel 0: player position (1).
        - Channel 1: car positions and their trails (-1 moving left, 1 moving right).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (10, 10),
        window_size: tuple = None,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        self.max_car_speed = -1
        self.cars = None
        self.player_row = None
        self.player_col = None
        self.player_row_old = None
        self.player_col_old = None

        # First channel for player pos
        # Second channel for cars pos and trail (-1 moving left, 1 moving right)
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 2), dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            "nop": 0,
            "up": 1,
            "down": 2,
        }

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

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_action = None

        self.player_row = self.n_rows - 1
        self.player_col = self.n_cols // 2
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # A car is denoted by (row, col, speed, direction, timer).
        # No car in the first and last row of the board.
        cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        speeds = self.np_random.integers(self.max_car_speed - 2, self.max_car_speed + 1, self.n_rows - 2)
        dirs = np.sign(self.np_random.uniform(-1, 1, self.n_rows - 2)).astype(np.int64)
        rows = np.arange(1, self.n_rows - 1)
        self.cars = [[r, c, s, d, 0] for r, c, s, d in zip(rows, cols, speeds, dirs)]

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def move(self, a):
        if a == DOWN:
            self.player_row = min(self.player_row + 1, self.n_rows - 1)
        elif a == UP:
            self.player_row = max(self.player_row - 1, 0)
        # elif a == RIGHT:
        #     self.player_col = min(self.player_col + 1, self.n_cols - 1)
        # elif a == LEFT:
            # self.player_col = max(self.player_col - 1, 0)
        elif a == NOP:
            pass
        else:
            raise ValueError("illegal action")

    def level_one(self):
        self.max_car_speed = -1
        self.reset()

    def level_up(self):
        self.max_car_speed = min(self.max_car_speed + 1, self.n_rows - 1)
        self.reset()

    def get_state(self):
        state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        state[self.player_row, self.player_col, 0] = 1
        for car in self.cars:
            row, col, speed, dir, timer = car
            if speed <= 0:
                if timer != speed:
                    speed = 0
                else:
                    speed = 1
            for step in range(speed + 1):
                state[row, (col + step * dir) % self.n_cols, 1] = dir
        return state

    def collision(self, row, col, action):
        return [row, col] == [self.player_row, self.player_col]

    def step(self, action: int):
        reward = 0.0
        terminated = False
        self.last_action = action

        # Move player
        self.player_row_old, self.player_col_old = self.player_row, self.player_col
        self.move(action)

        # Move cars
        for car in self.cars:
            row, col, speed, dir, timer = car
            if speed <= 0:
                if timer != speed:
                    car[4] -= 1
                    # Check if player moved on car that is not moving
                    if self.collision(row, col, action):
                        terminated = True
                        self.level_one()
                        break
                    continue
                else:
                    car[4] = 0
                    speed = 1
            for step in range(speed):
                col = (col + dir) % self.n_cols
                if self.collision(row, col, action):
                    terminated = True
                    self.level_one()
                    break
            car[1] = col

        # Win
        if self.player_row == 0:
            reward = 1.0
            self.level_up()

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
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
        else:
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

        assert self.window_surface is not None, "Something went wrong with pygame."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        def draw_tile(row, col, color):
            pos = (col * self.tile_size[0], row * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            pygame.draw.rect(self.window_surface, color, rect)

        # Draw cars and their trail
        for car in self.cars:
            row, col, speed, dir, timer = car
            draw_tile(row, col, RED)
            if speed <= 0:
                if timer != speed:
                    continue
                else:
                    speed = 1
            for step in range(max(0, speed)):
                col = (col - dir) % self.n_cols  # Backward for trail
                draw_tile(row, col, PALE_RED)

        # Draw player
        draw_tile(self.player_row, self.player_col, GREEN)

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
