import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2
SHOT = 3

RED = (255, 0, 0)  # submarine
PALE_RED = (255, 155, 155)  # submarine trail
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)


class SpaceInvaders(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        size: tuple = (10, 10),
        aliens_rows: int = 3,
        levels: int = 3,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        self.aliens_rows = aliens_rows

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        assert (  # One empty row before and after the aliens, one row for the player
            self.aliens_rows + 3 < self.n_rows
        ), f"cannot fit {aliens_rows} alien rows in a board with {self.n_rows} rows"

        # First channel for player position.
        # Second channel for aliens (-1 moving left, 1 moving right).
        # Third channel for bullets (-1 moving up, 1 moving down).
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3), dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.aliens = np.zeros((self.n_rows, self.n_cols), dtype=np.int64)
        self.aliens_dir = None
        self.leftmost_alien = None
        self.rightmost_alien = None
        self.bottom_alien = None
        self.aliens_move_down = None
        self.player_pos = None
        self.last_action = None

        # These two variable control the ball speed: when aliens_timesteps == aliens_delay,
        # the ball moves. A delay of 1 means that the player moves x2 times
        # faster than the ball.
        # When difficulty increases, aliens_delay decreases and can become negative.
        # Negative delay means that the ball is faster than the player.
        self.aliens_delay_levels = np.arange(levels - 1, -1, -1) - levels // 2
        self.level = 0
        self.aliens_delay = self.aliens_delay_levels[self.level]
        self.aliens_timesteps = self.aliens_delay

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

    def get_state(self):
        state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        state[..., 1] = self.aliens
        state[self.player_pos[0], self.player_pos[1], 1] = 1
        return state

    def level_one(self):
        self.level = 0
        self.reset()

    def level_up(self):
        self.level = min(self.level + 1, len(self.aliens_delay_levels - 1))
        self.reset()

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.player_pos = [
            self.n_rows - 1,
            self.np_random.integers((self.n_cols)),
        ]
        self.aliens_delay = self.aliens_delay_levels[self.level]
        self.aliens_timesteps = self.aliens_delay
        self.aliens[:] = 0
        self.aliens_dir = 1 if self.np_random.random() < 0.5 else -1
        self.aliens[1 : self.aliens_rows + 1, :] = self.aliens_dir
        self.aliens[:, 0] = 0  # First and last column are empty
        self.aliens[:, -1] = 0
        self.leftmost_alien = 1
        self.rightmost_alien = self.n_cols - 2
        self.bottom_alien = self.aliens_rows
        self.aliens_move_down = False
        self.last_action = None

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _step(self, action: int):
        self.last_action = action
        terminated = False
        reward = 0.0

        # Move player
        if action == NOP:
            pass
        elif action == LEFT:
            self.player_pos[1] = max(self.player_pos[1] - 1, 0)
        elif action == RIGHT:
            self.player_pos[1] = min(self.player_pos[1] + 1, self.n_cols - 1)
        elif action == SHOT:
            pass
        else:
            raise ValueError("illegal action")

        # Check if it's time to move the aliens
        if self.aliens_delay > 0:
            if self.aliens_timesteps != self.aliens_delay:
                self.aliens_timesteps += 1
                return self.get_state(), reward, terminated, False, {}
            else:
                self.aliens_timesteps = 0
                aliens_steps = 1
        else:
            aliens_steps = abs(self.aliens_delay) + 1

        for steps in range(aliens_steps):
            if self.aliens_move_down:
                self.aliens = np.roll(self.aliens, 1, 0)  # Move down
                self.bottom_alien += 1
                self.aliens_move_down = False
            else:
                self.aliens = np.roll(self.aliens, self.aliens_dir, 1)
                self.leftmost_alien -= self.aliens_dir
                self.rightmost_alien -= self.aliens_dir
                if self.leftmost_alien == 0 or self.rightmost_alien == self.n_cols - 1:
                    self.aliens_dir *= -1  # Change direction
                    self.aliens_move_down = True

        # Win or game over
        if self.aliens.sum() == 0:
            terminated = True
            self.level_one()
        elif self.bottom_alien == self.n_rows - 1:
            if self.level == len(self.aliens_delay_levels):
                terminated = True
            else:
                self.level_up()

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

        def draw_tile(row, col, color):
            pos = (col * self.tile_size[0], row * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            pygame.draw.rect(self.window_surface, color, rect)

        # Draw aliens
        for x in range(self.bottom_alien - self.aliens_rows + 1, self.bottom_alien + 1):
            for y in range(self.n_cols):
                if self.aliens[x, y]:
                    draw_tile(x, y, RED if self.aliens_dir == 1 else PALE_RED)

        # Draw alien bullets
        # draw_tile(self.aliens_pos[0], self.aliens_pos[1], BLUE)

        # Draw player bullets
        # draw_tile(self.aliens_pos[0], self.aliens_pos[1], BLUE)

        # Draw player
        draw_tile(self.player_pos[0], self.player_pos[1], GREEN)

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
