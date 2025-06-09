import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

RED = (255, 0, 0)
PINK = (255, 155, 155)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

# add ramp difficulty
# randomize colors
# this is different from minatar: there cars have slower speeds (for example, a car can take
# 2 frames to move) and it's much easier. It uses different shades of color to denote spees



class Freeway(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (8, 8),
        max_car_speed: int = None,
        side_movements: bool = True,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        if max_car_speed is None:
            max_car_speed = self.n_cols // 2
        self.max_car_speed = max_car_speed

        # ensure the game is winnable
        assert (
            self.max_car_speed <= self.n_cols // 2
        ), f"max car speed ({max_car_speed}) higher than half road width ({self.n_cols})"

        self.board = None
        self.car_cols = None
        self.chicken_id = max_car_speed + 1
        self.chicken_row = None
        self.chicken_col = None

        self.action_space = gym.spaces.Discrete(5 if side_movements else 3)
        self.observation_space = gym.spaces.Box(
            -max_car_speed,
            max_car_speed + 1,  # +1 for the chicken ID
            (self.n_rows, self.n_cols),
            dtype=np.int64,
        )

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
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
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=np.int64)

        self.chicken_row = self.n_rows - 1
        self.chicken_col = self.n_cols // 2
        self.board[self.chicken_row, self.chicken_col] = self.chicken_id

        # no car in first and last row
        self.car_cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        for i, car_col in enumerate(self.car_cols):
            speed = self.np_random.integers(1, self.max_car_speed + 1)
            direction = np.sign(float(i >= self.n_rows // 2) - 0.5).astype(np.int64)
            self.board[i + 1, car_col] = direction * speed

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.board.copy(), {}

    def move(self, a):
        if a == LEFT:
            self.chicken_col = max(self.chicken_col - 1, 0)
        elif a == DOWN:
            self.chicken_row = min(self.chicken_row + 1, self.n_rows - 1)
        elif a == RIGHT:
            self.chicken_col = min(self.chicken_col + 1, self.n_cols - 1)
        elif a == UP:
            self.chicken_row = max(self.chicken_row - 1, 0)
        elif a == NOP:
            pass
        else:
            raise ValueError("illegal action")

    def step(self, action: int):
        reward = 0.0
        terminated = False
        self.last_action = action

        # Move chicken but don't update the board yet
        self.move(action)

        # Move cars
        for i in range(len(self.car_cols)):
            car_col = self.car_cols[i]
            speed = self.board[i + 1, car_col]
            direction = np.sign(speed).astype(np.int64)
            self.board[i + 1, car_col] = 0  # remove car from current position
            for car_step in range(abs(speed)):
                car_col = (car_col + direction) % self.n_cols  # move car
                if [i + 1, car_col] == [self.chicken_row, self.chicken_col]:  # check for collision with chicken
                    self.chicken_row = self.n_rows - 1  # send chicken back to beginning
                    terminated = True
                    break
            self.car_cols[i] = car_col
            self.board[i + 1, car_col] = speed  # place car back on board

        # Check for winning condition
        if self.chicken_row == 0:
            reward = 1.0
            terminated = True

        # Finally, place chicken back on the board
        self.board[self.chicken_row, self.chicken_col] = self.chicken_id

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.board.copy(), reward, terminated, False, {}

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
                pygame.display.set_caption("NewFreeway")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None, "Something went wrong with pygame."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw cars and their trail
        for i in range(len(self.car_cols)):
            car_col = self.car_cols[i]
            speed = self.board[i + 1, car_col]
            direction = np.sign(speed).astype(np.int64)

            pos = (car_col * self.tile_size[0], (i + 1) * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            pygame.draw.rect(self.window_surface, RED, rect)

            for car_step in range(abs(speed)):
                car_col = (car_col - direction) % self.n_cols  # move backward to track car trail
                pos = (car_col * self.tile_size[0], (i + 1) * self.tile_size[1])
                rect = pygame.Rect(pos, self.tile_size)
                pygame.draw.rect(self.window_surface, PINK, rect)

        # Draw chicken
        pos = (
            self.chicken_col * self.tile_size[0],
            self.chicken_row * self.tile_size[1],
        )
        rect = pygame.Rect(pos, self.tile_size)
        pygame.draw.rect(self.window_surface, GREEN, rect)

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
