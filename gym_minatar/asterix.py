import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

BLACK = (0, 0, 0)
RED = (255, 0, 0)  # enemy
PALE_RED = (255, 155, 155)  # enemy trail
GREEN = (0, 255, 0)  # player
BLUE = (0, 0, 255)  # treasure
CYAN = (0, 255, 255)  # treasure trail

class Asterix(gym.Env):
    """
    The player moves on a grid and must collect treasures while avoiding enemies.
    - The player can move left/right/up/down or not move at all.
    - Enemies and treasures move horizontally with variable speed and direction.
    - When enemies and treasures leave the screen, some time must pass before a
      new random entity (enemy or treasure spawns) in the same row.
    - The player receives a reward for collecting treasures.
    - The game ends if the player is hit by an enemy.
    - The environment increases in difficulty over time (entities move faster
      and respawn sooner).
    - The observation space is a 3-channel grid with 0s for empty tiles, and 1 or -1
      for information about the game entities:
        - Channel 0: player position (1).
        - Channel 1: enemies and their trails (-1 moving left, 1 moving right).
        - Channel 2: treasures and their trails (-1 moving left, 1 moving right).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: tuple = (10, 10),
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        self.difficulty_timer = 0
        self.difficulty_increase_steps = 100
        self.cooldown = 3
        self.max_entity_speed = 1
        self.entities = None
        self.player_row = None
        self.player_col = None
        self.player_row_old = None
        self.player_col_old = None

        # First channel for player position.
        # Second channel for enemies position and their trail.
        # Third channel for treasures position and their trail.
        # For moving entities, -1 means movement to the left, +1 to the right.
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3), dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(5)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "down": 3,
            "up": 4,
        }

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
        state[self.player_row, self.player_col, 0] = 1
        for entity in self.entities:
            row, col, speed, dir, is_tres, timer, cooldown = entity
            if col is None:
                break
            if speed <= 0:
                if timer != speed:
                    speed = 0
                else:
                    speed = 1
            for step in range(speed + 1):
                if not 0 <= col - step * dir < self.n_cols:
                    break
                state[row, col - step * dir, 2 if is_tres else 1] = dir
        return state

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_action = None

        self.difficulty_timer = 0
        self.player_row = self.n_rows - 1
        self.player_col = self.n_cols // 2
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # Entries are denoted by (row, col, speed, direction, is_treasure, timer, cooldown).
        # Timer is for entries with negative speed (they move slower than the player).
        # Cooldown is for respawing.
        # First and last row of the board are empty.
        cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        speeds = self.np_random.integers(self.max_entity_speed - 2, self.max_entity_speed + 1, self.n_rows - 2)
        dirs = np.sign(self.np_random.uniform(-1, 1, self.n_rows - 2)).astype(np.int64)
        rows = np.arange(1, self.n_rows - 1)
        is_tres = self.np_random.random(self.n_rows - 2) < 1.0 / 3.0
        self.entities = [[r, c, s, d, i, 0, -1] for r, c, s, d, i in zip(rows, cols, speeds, dirs, is_tres)]

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

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

    def level_one(self):
        self.difficulty_timer = 0
        self.max_entity_speed = 2
        self.cooldown = 3

    def level_up(self):
        self.difficulty_timer = 0
        self.max_entity_speed = min(self.max_entity_speed + 1, self.n_rows - 1)
        self.cooldown = max(self.cooldown - 1, 0)

    def despawn(self, entity):
        entity[1] = None
        entity[6] = self.cooldown

    def respawn(self, entity):
        speed = self.np_random.integers(self.max_entity_speed - 2, self.max_entity_speed + 1)
        if self.np_random.random() < 0.5:
            col = 0
            dir = 1
        else:
            col = self.n_cols - 1
            dir = -1
        is_tres = self.np_random.random() < 1.0 / 3.0
        entity[1] = col
        entity[2] = speed
        entity[3] = dir
        entity[4] = is_tres
        entity[5] = 0
        entity[6] = self.cooldown

    def collision(self, row, col, action):
        # Must check horizontal movement, otherwise the player may "step over"
        # an entity and collision won't be detected
        return (
            ([row, col] == [self.player_row, self.player_col]) or
            (action in [LEFT, RIGHT] and ([row, col] == [self.player_row_old, self.player_col_old]))
        )

    def step(self, action: int):
        reward = 0.0
        terminated = False
        self.last_action = action

        self.difficulty_timer += 1
        if self.difficulty_timer == self.difficulty_increase_steps:
            self.level_up()

        # Move player
        self.player_row_old, self.player_col_old = self.player_row, self.player_col
        self.move(action)

        # Move enemies and treasures
        for entity in self.entities:
            row, col, speed, dir, is_tres, timer, cooldown = entity

            # Check if the entity is out of bounds, and if so check if it's time to respawn
            if col == None:
                cooldown -= 1
                entity[6] = cooldown
                if cooldown > 0:
                    continue
                else:
                    self.respawn(entity)
                    continue

            # If the speed is negative, check if the entity has waited enough before moving it
            if speed <= 0:
                if timer != speed:
                    entity[5] -= 1
                    continue
                else:
                    entity[5] = 0
                    speed = 1

            # Finally move the entity
            for step in range(speed):
                col += dir
                entity[1] = col
                if not 0 <= col < self.n_cols:
                    self.despawn(entity)
                    break
                if self.collision(row, col, action):
                    if is_tres:
                        self.despawn(entity)
                        reward = 1.0
                        break
                    else:
                        terminated = True
                        self.level_one()
                        self.reset()
                        break

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

        # Draw entities and their trail
        for entity in self.entities:
            row, col, speed, dir, is_tres, timer, cooldown = entity
            if col == None:
                continue
            draw_tile(row, col, BLUE if is_tres else RED)
            if speed <= 0:
                if timer != speed:
                    continue
                else:
                    speed = 1
            for step in range(max(0, speed)):
                col -= dir
                if not 0 <= col < self.n_cols:
                    break
                draw_tile(row, col, CYAN if is_tres else PALE_RED)

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
