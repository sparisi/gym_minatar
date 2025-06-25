import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

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

class Asterix(Game):
    """
    The player moves on a grid and must collect treasures while avoiding enemies.
    - The player can move left/right/up/down or not move at all.
    - Enemies and treasures move horizontally with variable speed and direction.
    - When enemies and treasures leave the screen, some time must pass before a
      new random entity (enemy or treasure) spawns in the same row.
    - The player receives a reward for collecting treasures.
    - The game ends if the player is hit by an enemy.
    - The environment increases in difficulty over time (entities move faster
      and respawn sooner).
    - The observation space is a 3-channel grid with 0s for empty tiles, and
      values in [-1, 1] for moving entities:
        - Channel 0: player position (1).
        - Channel 1: enemies and their trails (-1 moving left, 1 moving right).
        - Channel 2: treasures and their trails (-1 moving left, 1 moving right).
        - Intermediate values in (-1, 1) denote the speed of entities moving slower
          than 1 tile per step.
    """

    def __init__(self, **kwargs):
        Game.__init__(self, **kwargs)

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"

        self.difficulty_timer = 0
        self.difficulty_increase_steps = 100
        self.cooldown = 3
        self.max_entity_speed = -1
        self.entities = None
        self.player_row = None
        self.player_col = None
        self.player_row_old = None
        self.player_col_old = None

        # First channel for player position.
        # Second channel for enemies position and their trail (-1 moving left, 1 moving right).
        # Third channel for treasures position and their trail (-1 moving left, 1 moving right).
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3),
        )
        self.action_space = gym.spaces.Discrete(5)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
        }

    def get_state(self):
        state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        state[self.player_row, self.player_col, 0] = 1
        for entity in self.entities:
            row, col, speed, dir, is_tres, timer, cooldown = entity
            if col is None:
                break
            if speed <= 0:
                if timer != speed:
                    speed_scaling = (timer - 0.5) / speed
                    state[row, col, 2 if is_tres else 1] = dir
                    if 0 <= col - dir < self.n_cols:
                        state[row, (col - dir), 2 if is_tres else 1] = dir * speed_scaling
                    continue
                else:
                    speed = 1
            for step in range(speed + 1):
                if not 0 <= col - step * dir < self.n_cols:
                    break
                state[row, col - step * dir, 2 if is_tres else 1] = dir
        return state

    def _reset(self, seed: int = None, **kwargs):
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
        self.max_entity_speed = 0
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
        # Must check old position, otherwise the player may "step over"
        # an entity and collision won't be detected
        return (
            ([row, col] == [self.player_row, self.player_col]) or
            (action in [LEFT, RIGHT] and ([row, col] == [self.player_row_old, self.player_col_old]))
        )

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
                    # Check if the player moved on an entity that is not moving
                    if self.collision(row, col, action):
                        if is_tres:
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
                    if is_tres:
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
            row, col, speed, dir, is_tres, timer, cooldown = entity
            if col == None:
                continue
            self.draw_tile(row, col, BLUE if is_tres else RED)
            if speed <= 0:
                if timer != speed:
                    if 0 <= col < self.n_cols:
                        col = (col - dir)  # Backward for trail
                        speed_scaling = (timer - 0.5) / speed
                        self.draw_tile(row, col, CYAN if is_tres else PALE_RED, scale=speed_scaling)
                    continue
                else:
                    speed = 1
            for step in range(max(0, speed)):
                col -= dir
                if not 0 <= col < self.n_cols:
                    break
                self.draw_tile(row, col, CYAN if is_tres else PALE_RED)

        # Draw player
        self.draw_tile(self.player_row, self.player_col, GREEN)
