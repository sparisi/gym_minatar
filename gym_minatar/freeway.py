import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

# Action IDs
NOP = 0
UP = 1
DOWN = 2

RED = (255, 0, 0)  # cars
PALE_RED = (255, 155, 155)  # car trails
GREEN = (0, 255, 0)  # player
BLACK = (0, 0, 0)


class Freeway(Game):
    """
    The player (chicken) starts at the bottom of the screen and must reach the top,
    crossing lanes of moving cars.
    - The player can move up/down or may not move at all.
    - Cars move horizontally at different (random) speeds and directions,
      wrapping around the screen.
    - The game ends when the player is hit by a car.
    - Each time the player reaches the top, a reward of +1 is given and the car
      speeds increase (still random, though).
    - The observation space is a 2-channel grid with 0s for empty tiles, and
      values in [-1, 1] for moving entities:
        - Channel 0: player position (1).
        - Channel 1: car positions and their trails (-1 moving left, 1 moving right).
        - Intermediate values in (-1, 1) denote the speed of entities moving slower
          than 1 tile per timestep.
    """

    def __init__(self, **kwargs):
        Game.__init__(self, **kwargs)

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        self.max_speed = -1
        self.cars = None
        self.player_row = None
        self.player_col = None
        self.player_row_old = None
        self.player_col_old = None

        # First channel for player position.
        # Second channel for cars position and trail (-1 moving left, 1 moving right).
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 2),
        )  # fmt: skip
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            "nop": 0,
            "up": 1,
            "down": 2,
        }

    def _reset(self, seed: int = None, **kwargs):
        self.player_row = self.n_rows - 1
        self.player_col = self.n_cols // 2
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # A car is denoted by (row, col, speed, direction, timer).
        # No car in the first and last row of the board.
        cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        speeds = self.np_random.integers(self.max_speed - 2, self.max_speed + 1, self.n_rows - 2)
        dirs = np.sign(self.np_random.uniform(-1, 1, self.n_rows - 2)).astype(np.int64)
        rows = np.arange(1, self.n_rows - 1)
        self.cars = [[r, c, s, d, 0] for r, c, s, d in zip(rows, cols, speeds, dirs)]

        return self.get_state(), {}

    def move(self, a):
        if a == DOWN:
            self.player_row = min(self.player_row + 1, self.n_rows - 1)
        elif a == UP:
            self.player_row = max(self.player_row - 1, 0)
        elif a == NOP:
            pass
        else:
            raise ValueError("illegal action")

    def level_one(self):
        self.max_speed = -1
        self._reset()

    def level_up(self):
        self.max_speed = min(self.max_speed + 1, self.n_rows - 1)
        self._reset()

    def get_state(self):
        state = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        state[self.player_row, self.player_col, 0] = 1
        for car in self.cars:
            row, col, speed, dir, timer = car
            if speed <= 0:
                if timer != speed:
                    speed_scaling = (timer - 0.5) / speed
                    state[row, (col - dir) % self.n_cols, 1] = dir * speed_scaling
                    state[row, (col) % self.n_cols, 1] = dir
                    continue
                else:
                    speed = 1
            for step in range(speed + 1):
                state[row, (col - step * dir) % self.n_cols, 1] = dir
        return state

    def collision(self, row, col, action):
        return [row, col] == [self.player_row, self.player_col]

    def _step(self, action: int):
        reward = 0.0
        terminated = False

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

        return self.get_state(), reward, terminated, False, {}

    def _render_board(self):
        import pygame

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw cars and their trail
        for car in self.cars:
            row, col, speed, dir, timer = car
            self.draw_tile(row, col, RED)
            if speed <= 0:
                if timer != speed:
                    col = (col - dir) % self.n_cols  # Backward for trail
                    speed_scaling = (timer - 0.5) / speed
                    self.draw_tile(row, col, PALE_RED, scale=speed_scaling)
                    continue
                else:
                    speed = 1
            for step in range(max(0, speed)):
                col = (col - dir) % self.n_cols
                self.draw_tile(row, col, PALE_RED)

        # Draw player
        self.draw_tile(self.player_row, self.player_col, GREEN)
