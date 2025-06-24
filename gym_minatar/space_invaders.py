import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2
SHOT = 3

BLACK = (0, 0, 0)
RED = (255, 0, 0)  # aliens moving right
PALE_RED = (255, 155, 155)  # aliens moving left
GREEN = (0, 255, 0)  # player
WHITE = (255, 255, 255)  # player bullet
YELLOW = (255, 255, 0)  # alien bullet


class SpaceInvaders(gym.Env):
    """
    The player controls a spaceship at the bottom of the screen that must shoot
    down waves of aliens.
    - The player moves left/right or not move at all, and can shoot.
    - Aliens move horizontally and descend when they hit the screen left/right
      edges.
        - Their speed increases as they descend: + 1 level for every time they
          move down for as many times as the initial number of aliens rows.
          For example, if at the beginning there are 3 aliens rows, the speed
          increases by 1 after they reach row 6, then 9, ... and so on.
        - Every level increases the speed by as many frames as the number of
          initial alien rows.
        - At their fastest, the aliens move as fast as the player.
    - A random alien shoots whenever possible (there is a cooldown time
      shared by all aliens).
      - Aliens bullet move as fast as the player, regardless of the aliens speed.
    - The player also must wait before it can shoot again.
    - The player receives a reward for destroying aliens with bullets.
    - The game ends if an alien reaches the bottom or the player is hit.
    - If the player destroys all aliens, the next level starts.
    - Difficulty increases with levels, making aliens starting closer to the player.
    - The observation space is a 3-channel grid with 0s for empty tiles, and 1 or -1
      for information about the game entities:
        - Channel 0: player position (1).
        - Channel 1: aliens (-1 moving left, 1 moving right).
        - Channel 2: bullets (-1 moving up, 1 moving down).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (10, 10),
        aliens_rows: int = 3,
        window_size: tuple = None,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        self.aliens_rows = aliens_rows

        assert self.aliens_rows > 0, f"aliens rows must be positive (received {self.n_rows})"
        # First two and last two columns must be empty at the beginning
        assert self.n_cols > 4, f"board too small ({self.n_cols} columns, minimum is 5)"
        # One row for the player, one for the aliens, one for moving aliens down once
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows, minimum is 3)"
        # One empty row below the aliens, one row for the player
        assert (
            self.n_rows >= self.aliens_rows + 2
        ), f"cannot fit {aliens_rows} alien rows in a board with {self.n_rows} rows"

        # First channel for player position.
        # Second channel for aliens (-1 moving left, 1 moving right).
        # Third channel for player bullets (-1, always moving up).
        # Fourh channel for aliens bullets (1, always moving down).
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 4), dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "shoot": 3,
        }

        self.state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self.aliens_dir = None
        self.bottom_alien = None  # Keep track of how much the aliens moved down for faster rendering
        self.aliens_move_down = None
        self.player_pos = None
        self.last_action = None
        self.player_shoot_cooldown = 3
        self.player_shoot_timer = 0
        self.alien_shoot_cooldown = 5
        self.alien_shoot_timer = 0
        self.starting_row = 0  # Denotes the current level of the game

        # These two variable control the aliens speed: when aliens_timesteps == aliens_delay,
        # the aliens move. A delay of 1 means that the player moves x2 times
        # faster than the aliens.
        # Speed increases as aliens move down (+1 for every aliens_rows down).
        # One speed level corresponds to a delay equal to the number of initial alien rows.
        # For example...
        # Min delay is 0, that is aliens move as fast as the player.
        self.aliens_delay_levels = np.arange(self.n_rows // self.aliens_rows - 1, -1, -1)
        self.aliens_delay_levels *= self.aliens_rows
        self.aliens_delay = self.aliens_delay_levels[0]
        self.aliens_timesteps = 0
        self.lowest_row_reached = self.bottom_alien

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
        return self.state.copy()

    def level_one(self):
        self.starting_row = 0
        self.reset()

    def level_up(self):
        self.starting_row = min(self.starting_row + 1, self.n_rows - self.aliens_rows - 1)
        self.reset()

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_action = None

        self.player_shoot_timer = 0
        self.alien_shoot_timer = 0
        self.state[:] = 0

        self.player_pos = [
            self.n_rows - 1,
            self.np_random.integers((self.n_cols)),
        ]
        self.state[self.player_pos[0], self.player_pos[1], 0] = 1

        self.aliens_delay = self.aliens_delay_levels[0]
        self.aliens_timesteps = 0

        self.aliens_dir = 1 if self.np_random.random() < 0.5 else -1
        self.state[self.starting_row : self.starting_row + self.aliens_rows, :, 1] = self.aliens_dir
        self.state[:, 0, 1] = 0  # First two and last two columns are empty
        self.state[:, 1, 1] = 0
        self.state[:, -2, 1] = 0
        self.state[:, -1, 1] = 0
        self.bottom_alien = self.starting_row + self.aliens_rows - 1
        self.lowest_row_reached = self.bottom_alien
        self.aliens_move_down = False

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def player_shoot(self):
        if self.player_shoot_timer > 0:
            return
        self.player_shoot_timer = self.player_shoot_cooldown
        # Row is the same as player's because the bullet will be moved up in the same turn
        self.state[self.player_pos[0], self.player_pos[1], 2] = -1

    def alien_shoot(self):
        if self.alien_shoot_timer > 0:
            return
        self.alien_shoot_timer = self.alien_shoot_cooldown
        where_aliens = np.argwhere(self.state[..., 1])
        who_shoots = where_aliens[self.np_random.choice(where_aliens.shape[0])]
        # Row is the same as player's because the bullet will be moved up in the same turn
        self.state[who_shoots[0], who_shoots[1], 3] = 1

    def _step(self, action: int):
        self.last_action = action
        terminated = False
        reward = 0.0

        # Cooldown
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1
        if self.alien_shoot_timer > 0:
            self.alien_shoot_timer -= 1
        else:
            self.alien_shoot()

        # Move player
        self.state[self.player_pos[0], self.player_pos[1], 0] = 0
        if action == NOP:
            pass
        elif action == LEFT:
            self.player_pos[1] = max(self.player_pos[1] - 1, 0)
        elif action == RIGHT:
            self.player_pos[1] = min(self.player_pos[1] + 1, self.n_cols - 1)
        elif action == SHOT:
            self.player_shoot()
        else:
            raise ValueError("illegal action")
        self.state[self.player_pos[0], self.player_pos[1], 0] = 1

        # Check if it's time to move the aliens
        if self.aliens_delay > 0:
            if self.aliens_timesteps < self.aliens_delay:
                self.aliens_timesteps += 1
                aliens_steps = 0
            else:
                self.aliens_timesteps = 0
                aliens_steps = 1
        else:
            aliens_steps = abs(self.aliens_delay) + 1

        bullets_moved = False
        def move_bullets():  # And check if player bullets hit aliens
            self.state[..., 2] = np.roll(self.state[..., 2], -1, 0)  # Player bullets up
            self.state[-1, :, 2] = 0  # Remove if rolled back to bottom
            self.state[..., 3] = np.roll(self.state[..., 3], 1, 0)  # Aliens bullets down
            self.state[0, :, 3] = 0  # Remove if rolled back to top
            alien_hits = self.state[..., 2] * self.state[..., 1] * self.aliens_dir * -1
            self.state[..., 1] *= (1 - alien_hits)
            self.state[..., 2] *= (1 - alien_hits)

        # Move aliens down/left/right
        for steps in range(aliens_steps):
            if self.aliens_move_down:
                # Move bullets before moving aliens down, or hits may not be detected
                if not bullets_moved:
                    move_bullets()
                    bullets_moved = True

                self.state[..., 1] = np.roll(self.state[..., 1], 1, 0)
                self.bottom_alien += 1

                # Make aliens faster, if they moved down enough
                self.lowest_row_reached = max(self.bottom_alien, self.lowest_row_reached)
                self.aliens_move_down = False
                delay_level = (self.lowest_row_reached + self.aliens_rows) // self.aliens_rows - 2
                delay_level = min(delay_level, len(self.aliens_delay_levels) - 1)  # Otherwise, error at bottom line (game over)
                self.aliens_delay = self.aliens_delay_levels[delay_level]
            else:
                self.state[..., 1] = np.roll(self.state[..., 1], self.aliens_dir, 1)
                if (
                    np.any(self.state[self.bottom_alien - self.aliens_rows + 1 : self.bottom_alien + 1, 0, 1] != 0) or
                    np.any(self.state[self.bottom_alien - self.aliens_rows + 1 : self.bottom_alien + 1, -1, 1] != 0)
                ):
                    self.state[..., 1] *= -1
                    self.aliens_dir *= -1  # Change direction
                    self.aliens_move_down = True

        # Move bullets only once per step
        if not bullets_moved:
            move_bullets()

        alien_left_rows = np.nonzero(np.any(self.state[..., 1], axis=1))[0]
        if len(alien_left_rows) > 0:
            self.bottom_alien = alien_left_rows.max()
        else:  # All aliens destroyed
            self.bottom_alien = None
            self.level_up()

        # Win or game over conditions
        if self.state[self.player_pos[0], self.player_pos[1], 3] != 0:  # Player hit
            terminated = True
            self.level_one()
        elif self.bottom_alien == self.player_pos[0]:  # Aliens reached the bottom
            terminated = True
            self.level_one()

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
                if self.state[x, y, 1]:
                    draw_tile(x, y, RED if self.aliens_dir == 1 else PALE_RED)

        # for x in range(self.bottom_alien - self.aliens_rows + 1, self.n_rows):
        for x in range(self.n_rows):
            for y in range(self.n_cols):
                if self.state[x, y, 3] == 1:
                    draw_tile(x, y, YELLOW)
                elif self.state[x, y, 2] == -1:
                    draw_tile(x, y, WHITE)

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
