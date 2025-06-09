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
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)

# add ramp difficulty
# randomize colors
# this is different from minatar: there enemies have slower speeds (for example, a enemy can take
# 2 frames to move) and it's much easier. It uses different shades of color to denote spees

 # +1 because first row is always empty


class Asterix(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (8, 8),
        max_speed: int = None,
        spawn_time: int = 3,
        **kwargs,
    ):
        assert spawn_time > 0, f"spawn time must be positive (received {spawn_time})"
        self.spawn_time = spawn_time

        self.n_rows, self.n_cols = size
        if max_speed is None:
            max_speed = self.n_cols // 2
        self.max_speed = max_speed

        # ensure the game is winnable
        assert (
            self.max_speed <= self.n_cols // 2
        ), f"max speed ({max_speed}) higher than half board width ({self.n_cols})"

        self.board = None
        self.entry_cols = None  # stores location of enemies / treasures
        self.is_treasure = None  # is_treasure[i] = True if entry_cols[i] is a treasure (False if enemy)
        self.player_id = max_speed + 1
        self.player_row = None
        self.player_col = None
        self.cooldowns = None

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            -max_speed,
            max_speed + 1,  # +1 for the player ID
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

        self.player_row = self.n_rows - 1
        self.player_col = self.n_cols // 2
        self.board[self.player_row, self.player_col] = self.player_id

        # First and last row are always empty
        self.cooldowns = np.full((self.n_rows - 2,), -1, dtype=np.int64)
        self.entry_cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        self.is_treasure = self.np_random.random(self.n_rows - 2) < 0.5
        for i, entry_col in enumerate(self.entry_cols):
            speed = self.np_random.integers(1, self.max_speed + 1)
            direction = np.sign(float(i >= self.n_rows // 2) - 0.5).astype(np.int64)
            self.board[i + 1, entry_col] = direction * speed

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.board.copy(), {}

    def move(self, a):
        if a == LEFT:
            self.player_col = max(self.player_col - 1, 0)
        elif a == DOWN:
            self.player_row = min(self.player_row + 1, self.n_rows - 1)
        elif a == RIGHT:
            self.player_col = min(self.player_col + 1, self.n_cols - 1)
        elif a == UP:
            self.player_row = max(self.player_row - 1, 0)
        elif a == NOP:
            pass
        else:
            raise ValueError("illegal action")

    def despawn(self, i):
        self.cooldowns[i] = self.spawn_time
        self.entry_cols[i] = -1

    def spawn(self, i):
        speed = self.np_random.integers(1, self.max_speed + 1)
        if self.np_random.random() < 0.5:
            col = 0
            direction = 1
        else:
            col = self.n_cols - 1
            direction = -1
        self.cooldowns[i] = -1
        self.entry_cols[i] = col
        self.is_treasure[i] = self.np_random.random() < 1.0 / 3.0
        self.board[i + 1, col] = speed * direction

    def check_collision(self, entry_idx, col):
        hit_car, hit_treasure = False, False
        if [entry_idx + 1, col] == [self.player_row, self.player_col]:
            if not self.is_treasure[entry_idx]:  # enemy sends player back to beginning
                self.player_row = self.n_rows - 1
                hit_car = True
            else:
                hit_treasure = True
                self.despawn(entry_idx)
        return hit_car, hit_treasure

    def step(self, action: int):
        reward = 0.0
        terminated = False
        self.last_action = action

        # Move the player but don't update the board yet
        self.move(action)

        # Move enemies and treasures
        for i in range(len(self.entry_cols)):
            # Check cooldown and spawn
            if self.cooldowns[i] > 0:  # still cooling down
                self.cooldowns[i] -= 1
                continue
            elif self.cooldowns[i] == 0:
                self.cooldowns[i] = -1
                self.spawn(i)
                hit_car, hit_treasure = self.check_collision(i, self.entry_cols[i])
                if hit_treasure:  # if the player is where an entry spawns, it may die or get the reward
                    reward = 1.0
                continue  # after spawn, do not move immediately

            # Move one step at the time
            entry_col = self.entry_cols[i]
            speed = self.board[i + 1, entry_col]
            direction = np.sign(speed).astype(np.int64)
            for entry_step in range(abs(speed)):
                self.board[i + 1, entry_col] = 0  # remove entry from current position
                entry_col = (entry_col + direction)
                if not (0 <= entry_col < self.n_cols):  # moving out of bounds
                    self.despawn(i)
                    break
                hit_car, hit_treasure = self.check_collision(i, entry_col)
                if hit_treasure:  # if the player is where an entry spawns, it may die or get the reward
                    reward = 1.0
                    break  # check_collision() already despawn treasures
                self.entry_cols[i] = entry_col
                self.board[i + 1, entry_col] = speed  # place entry back on board

        # Finally, place the player back on the board
        self.board[self.player_row, self.player_col] = self.player_id

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

        # Draw enemies/treasures and their trail
        for i in range(len(self.entry_cols)):
            entry_col = self.entry_cols[i]
            speed = self.board[i + 1, entry_col]
            direction = np.sign(speed).astype(np.int64)

            pos = (entry_col * self.tile_size[0], (i + 1) * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            color = BLUE if self.is_treasure[i] else RED
            pygame.draw.rect(self.window_surface, color, rect)

            for entry_step in range(abs(speed)):
                entry_col = (entry_col - direction) % self.n_cols  # move backward to track trail
                pos = (entry_col * self.tile_size[0], (i + 1) * self.tile_size[1])
                rect = pygame.Rect(pos, self.tile_size)
                color = CYAN if self.is_treasure[i] else PINK
                pygame.draw.rect(self.window_surface, color, rect)

        # Draw player
        pos = (
            self.player_col * self.tile_size[0],
            self.player_row * self.tile_size[1],
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
