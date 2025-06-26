import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

# Entity IDs
ENEMY = 1
TREASURE = 2

BLACK = (0, 0, 0)
RED = (255, 0, 0)  # enemy
PALE_RED = (255, 155, 155)  # enemy trail
GREEN = (0, 255, 0)  # player
BLUE = (0, 0, 255)  # treasure
CYAN = (0, 255, 255)  # treasure trail


class Asterix(Game):
    """
    The player moves on a grid and must collect treasures while avoiding enemies.
    - The player can move left/right/up/down, or stand still.
    - Enemies and treasures move horizontally with variable speed and direction.
      - Each entity's speed is randomly selected at the beginning in
        [self.speed - self.speed_range, self.speed].
    - When enemies and treasures leave the screen, some time must pass before a
      new one (randomly either an enemy or a treasure) spawns in the same row.
    - The player receives +1 for collecting treasures.
    - The game ends if the player is hit by an enemy.
    - The environment increases in difficulty over time (speed increases
      by 1, respawn time decreases by 1).
    - The observation space is a 3-channel grid with 0s for empty tiles, and
      values in [-1, 1] for game entities:
        - Channel 0: player position (1).
        - Channel 1: enemies and their trails (-1 moving left, 1 moving right).
        - Channel 2: treasures and their trails (-1 moving left, 1 moving right).
        - Intermediate values in (-1, 1) denote the speed of entities moving slower
          than 1 tile per timestep.
    """

    def __init__(self, **kwargs):
        Game.__init__(self, **kwargs)

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"

        self.difficulty_timer = 0
        self.difficulty_increase_steps = 100

        # Please see freeway.py for more details about these variables
        self.init_speed = 0
        self.max_speed = self.n_cols - 3
        self.speed = self.init_speed
        self.speed_range = 2  # Entity speed will be in [self.speed - self.speed_range, self.speed]
        n_slow_speeds = self.init_speed - self.speed_range - 1
        self.slow_speed_bins = np.arange(n_slow_speeds, 0) / n_slow_speeds

        self.init_cooldown = 3
        self.cooldown = self.init_cooldown

        self.entities = None
        self.player_row = None
        self.player_col = None
        self.player_row_old = None
        self.player_col_old = None
        self.treasure_prob = 1.0 / 3.0

        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3),
        )  # fmt: skip
        self.action_space = gym.spaces.Discrete(5)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
        }

    def get_state(self):
        state = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        state[self.player_row, self.player_col, 0] = 1
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown = entity
            if col is None:
                continue
            state[row, col, id] = dir  # Entity
            speed_scaling = self.slow_speed_bins[max(timer - speed, 0)]
            for step in range(1, max(1, speed) + 1):  # Speed trail
                if not 0 <= col - step * dir < self.n_cols:
                    break
                state[row, (col - step * dir), id] = dir * speed_scaling
        return state

    def _reset(self, seed: int = None, **kwargs):
        self.difficulty_timer = 0
        self.player_row = self.n_rows - 1
        self.player_col = self.n_cols // 2
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # Entities are denoted by (row, col, speed, direction, is_treasure, timer, cooldown).
        # Timer is for entities with negative speed (they move slower than the player).
        # Cooldown is for respawing.
        # First and last row of the board are empty.
        cols = self.np_random.integers(0, self.n_cols, self.n_rows - 2)
        speeds = self.np_random.integers(self.speed - self.speed_range, self.speed + 1, self.n_rows - 2)
        dirs = np.sign(self.np_random.uniform(-1, 1, self.n_rows - 2)).astype(np.int64)
        rows = np.arange(1, self.n_rows - 1)
        id = np.full((self.n_rows - 2,), ENEMY)
        id[self.np_random.random(self.n_rows - 2) < self.treasure_prob] = TREASURE
        self.entities = [
            [r, c, s, d, i, 0, -1]
            for r, c, s, d, i in zip(rows, cols, speeds, dirs, id)
        ]

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
        self.speed = self.init_speed
        self.cooldown = self.init_cooldown

    def level_up(self):
        self.difficulty_timer = 0
        self.speed = min(self.speed + 1, self.max_speed)
        self.cooldown = max(self.cooldown - 1, 0)

    def despawn(self, entity):
        entity[1] = None
        entity[6] = self.cooldown

    def respawn(self, entity):
        speed = self.np_random.integers(self.speed - self.speed_range, self.speed + 1)
        if self.np_random.random() < 0.5:
            col = 0
            dir = 1
        else:
            col = self.n_cols - 1
            dir = -1
        id = TREASURE if self.np_random.random() < self.treasure_prob else ENEMY
        entity[1] = col
        entity[2] = speed
        entity[3] = dir
        entity[4] = id
        entity[5] = 0
        entity[6] = self.cooldown

    def collision(self, row, col, action):
        static_collision = [row, col] == [self.player_row, self.player_col]
        # Without this check, the player may "step over" an entity and collision won't be detected
        movement_collision = (
            action in [LEFT, RIGHT] and
            [row, col] == [self.player_row_old, self.player_col_old]
        )
        return static_collision or movement_collision

    def _step(self, action: int):
        reward = 0.0
        terminated = False

        self.difficulty_timer += 1
        if self.difficulty_timer == self.difficulty_increase_steps:
            self.level_up()

        # Move player
        self.player_row_old, self.player_col_old = self.player_row, self.player_col
        self.move(action)

        # Move enemies and treasures
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown = entity

            # Check if the entity is out of bounds, and if so check if it's time to respawn
            if col is None:
                cooldown -= 1
                entity[6] = cooldown
                if cooldown > 0:
                    continue
                else:
                    self.respawn(entity)
                    continue

            # If the speed is negative, check if the entity has waited enough before moving it
            if speed <= 0:
                if timer > speed:
                    entity[5] -= 1
                    # Check if the player moved on an entity that is not moving
                    if self.collision(row, col, action):
                        if id == TREASURE:
                            self.despawn(entity)
                            reward = 1.0
                            break
                        else:
                            terminated = True
                            self.level_one()
                            self._reset()
                            break
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
                    if id == TREASURE:
                        self.despawn(entity)
                        reward = 1.0
                        break
                    else:
                        terminated = True
                        self.level_one()
                        self._reset()
                        break

        return self.get_state(), reward, terminated, False, {}

    def _render_board(self):
        import pygame

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw entities and their trail
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown = entity
            if col is None:
                continue

            if id == TREASURE:
                color_main = BLUE
                color_trail = CYAN
            else:
                color_main = RED
                color_trail = PALE_RED

            self.draw_tile(row, col, color_main)
            speed_scaling = self.slow_speed_bins[max(timer - speed, 0)]
            for step in range(max(1, speed)):
                col -= dir
                if not 0 <= col < self.n_cols:
                    break
                self.draw_tile(row, col, color_trail, speed_scaling)

        # Draw player
        self.draw_tile(self.player_row, self.player_col, GREEN)
