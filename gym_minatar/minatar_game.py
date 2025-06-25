import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

class Game(gym.Env):
    """
    Base class of Gym-MinAtar games.
    Each game implements its own _render_board(), _step(), _reset().
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
        obs, info = self._reset(seed, **kwargs)
        self.last_action = None
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, info

    def _reset(self, seed: int = None, **kwargs):
        pass


    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        self.last_action = action
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _step(self, action: int):
        pass


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

        self._render_board()

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

    def _render_board(self):
        pass

    def draw_tile(self, row, col, color, scale=1.0):
        import pygame
        pos = (col * self.tile_size[0], row * self.tile_size[1])
        rect = pygame.Rect(pos, self.tile_size)
        if scale != 1.0:
            rect = rect.scale_by(scale)
        pygame.draw.rect(self.window_surface, color, rect)

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
