import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
SHOOT = 5

# Entity IDs
FISH = 1
SUBMARINE = 2
DIVER = 3

BLACK = (0, 0, 0)
GRAY = (100, 100, 100)  # surface
RED = (255, 0, 0)  # submarine
PALE_RED = (255, 155, 155)  # submarine trail
YELLOW = (255, 255, 0)  # submarine bullet
PURPLE = (102, 51, 153)  # fish
PALE_PURPLE = (200, 155, 255)  # fish trail
BLUE = (0, 0, 255)  # diver
CYAN = (0, 255, 255)  # diver trail
PALE_CYAN = (200, 255, 255)  # diver gauge
GREEN = (0, 255, 0)  # player front
PALE_GREEN = (200, 255, 200)  # player back
WHITE = (255, 255, 255)  # player bullet
PALE_YELLOW = (255, 255, 200)  # player oxygen

# Bullets don't have a trail because their direction can be inferred from the
# submarine / player position. Bullets travel faster than the entity that shot them,
# so a bullet to the left of the player can only be going left.


class Seaquest(Game):
    """
    The player controls a submarine that can move and shoot, collecting divers
    and avoiding enemies.
    - The player occupies two horizontal tiles and can move left/right/up/down,
      or stand still.
        - If the player is facing left (or right) and moves right (or left),
          it turns around (front and back swap).
    - The player can also shoot in front of itself.
        - Shooting has a cooldown.
        - The bullet moves twice as fast as the player.
        - Player bullets destroy enemies on collision, giving +1 to the player.
    - Enemies and divers move at different speeds and leave a trail behind them
      to infer their direction.
      - Each entity's speed is randomly selected at the beginning in
        [self.speed - self.speed_range, self.speed].
    - Enemies can be fish or submarine.
        - Submarines shoot, fishes don't.
        - As soon as their bullet leaves the screen, they shoot a new one.
        - Their bullet moves 1 frames faster than them.
        - Player and enemies bullets don't collide.
        - If the player is hit by the enemies bullets, it dies.
        - If a submarine is hit by the player bullets, its bullet disappears as well.
    - The player can collect divers (up to 6) on collision.
    - The player has an oxygen reserve (gauge at the bottom left) that depletes
      over time. When it's empty, the game ends.
    - The player can emerge by moving to the top.
      - If the player is carrying 6 divers, it gets a as many points as the
        amount of oxygen left in the gauge.
      - If the player is carrying at least 1 divers, its oxygen is fully restored
        but it doesn't get any point.
      - If the player is not carrying any diver, the game ends.
      - The player can stay at the surface as long as it wants without depleting
        oxygen.
    - When the player submerges again, the difficulty level increases
      (speed increases by 1, respawn time decreases by 1).
    - The observation space is a 6-channel grid with 0s for empty tiles, and
      values in [-1, 1] for moving entities:
        - Channel 0: player position and bullets (-1 moving left, 1 moving right).
        - Channel 1: fishes and their trails (-1 moving left, 1 moving right).
        - Channel 2: submarines, bullets, and their trails (-1 moving left, 1 moving right).
        - Channel 3: divers and their trails (-1 moving left, 1 moving right).
        - Channel 4: oxygen gauge (1s at the bottom row).
        - Channel 5: divers carried gauge (1s at the bottom row).
        - Intermediate values in (-1, 1) denote enemies and divers speed
          when they move slower than 1 tile per timestep.
    """

    def __init__(self, **kwargs):
        Game.__init__(self, **kwargs)

        # Last row is for oxygen and divers gauge, first row is the surface
        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"

        # Entities are denoted by (row, col, speed, direction, id, timer, cooldown, bullet_col).
        # Timer is for entities with negative speed (they move slower than the player).
        # Cooldown is for respawing.
        # Bullet column is None except for submarines.
        self.entities = None

        # Please see freeway.py for more details about these variables
        self.init_speed = -2
        self.speed = self.init_speed
        self.speed_range = 2  # Entity speed will be in [self.speed - self.speed_range, self.speed]
        n_partial_speeds = self.init_speed - self.speed_range - 1
        self.speed_chunks = np.arange(n_partial_speeds, 0) / n_partial_speeds

        self.init_spawn_cooldown = 3
        self.spawn_cooldown = self.init_spawn_cooldown

        self.player_row = None
        self.player_col = None
        self.player_dir = None
        self.player_row_old = None
        self.player_col_old = None
        self.player_bullets = None
        self.shoot_cooldown = 3
        self.shoot_timer = 0
        self.oxygen_max = 20
        self.oxygen_decay = 10  # Number of steps before oxygen goes down by 1
        self.divers_carried_max = 6
        self.oxygen = self.oxygen_max
        self.oxygen_counter = 0
        self.divers_carried = 0
        # 50% fish, 25% submarines, 25% divers (first 0% is because ID start from 1)
        self.spawn_probs = np.array([0.0, 0.5, 0.15, 0.35])
        self.spawn_probs_cdf = self.spawn_probs.cumsum(0)

        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 6),
        )  # fmt: skip
        self.action_space = gym.spaces.Discrete(6)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
            "shoot": 5,
        }

    def get_state(self):
        state = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        state[self.player_row, self.player_col, 0] = self.player_dir
        state[self.player_row, self.player_col - self.player_dir, 0] = self.player_dir
        for bullet in self.player_bullets:
            row, col, dir = bullet
            state[row, col, 0] = dir
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown, b_col = entity
            if col is None:
                continue
            if b_col is not None and 0 <= b_col < self.n_cols:  # Bullet
                state[row, b_col, id] = dir
            state[row, col, id] = dir  # Entity
            speed_scaling = self.speed_chunks[max(timer - speed, 0)]
            for step in range(1, max(1, speed) + 1):  # Speed trail
                if not 0 <= col - step * dir < self.n_cols:
                    break
                state[row, (col - step * dir), id] = dir * speed_scaling

        percentage_full = self.divers_carried / self.divers_carried_max
        n_fill = int(self.n_cols * percentage_full)
        if percentage_full > 0:
            n_fill = max(1, n_fill)  # At least one 1 when the player is carrying 1 diver
        for i in range(n_fill):
            state[-1, i, 4] = 1

        percentage_full = self.oxygen / self.oxygen_max
        n_fill = int(self.n_cols * percentage_full)
        if percentage_full > 0:
            n_fill = max(1, n_fill)  # At least one 1 when the player has oxygen left
        for i in range(n_fill):
            state[-1, i, 5] = 1

        return state

    def level_one(self):
        self.speed = self.init_speed
        self.spawn_cooldown = self.init_spawn_cooldown

    def level_up(self):
        self.speed = min(self.speed + 1, self.n_rows - 1)
        self.spawn_cooldown = max(self.spawn_cooldown - 1, 0)

    def _reset(self, seed: int = None, **kwargs):
        self.shoot_timer = 0
        self.player_bullets = []  # Dynamic list because there can be many bullets on the board
        self.player_row = self.n_rows - 2
        self.player_dir = 1 if self.np_random.random() < 0.5 else -1
        if self.player_dir == 1:  # Facing right
            self.player_col = self.np_random.integers(1, self.n_cols)
        else:  # Facing left
            self.player_col = self.np_random.integers(0, self.n_cols - 1)
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # First and last row of the board are empty.
        # Board is empty at the beginning (all values are None) with
        # random cooldowns, so entities will spawn little by little.
        rows = np.arange(1, self.n_rows - 1)
        cdowns = self.np_random.integers(0, self.spawn_cooldown, self.n_rows - 2)
        self.entities = [
            [r, None, None, None, None, 0, cd, None]
            for r, cd in zip(rows, cdowns)
        ]

        self.oxygen = self.oxygen_max
        self.oxygen_counter = 0
        self.divers_carried = 0

        return self.get_state(), {}

    def shoot(self):
        if self.shoot_timer > 0:
            return
        self.shoot_timer = self.shoot_cooldown
        col = self.player_col + self.player_dir
        if not 0 <= col < self.n_cols:
            return
        if self.collision_with_entity(self.player_row_old, col):
            return
        self.player_bullets.append([self.player_row, col, self.player_dir])

    def move(self, a):
        if a == LEFT:
            self.player_col = max(self.player_col - 1, 0)
            if self.player_dir == 1:  # Turn
                self.player_dir = -1
        elif a == DOWN:
            self.player_row = min(self.player_row + 1, self.n_rows - 2)
        elif a == RIGHT:
            self.player_col = min(self.player_col + 1, self.n_cols - 1)
            if self.player_dir == -1:
                self.player_dir = 1
        elif a == UP:
            self.player_row = max(self.player_row - 1, 0)
        elif a in [NOP, SHOOT]:
            pass
        else:
            raise ValueError("illegal action")

    def despawn(self, entity):
        entity[1] = None
        entity[6] = self.spawn_cooldown
        if entity[4] == SUBMARINE:
            entity[7] = None

    def respawn(self, entity):
        speed = self.np_random.integers(self.speed - self.speed_range, self.speed + 1)
        if self.np_random.random() < 0.5:
            col = 0
            dir = 1
        else:
            col = self.n_cols - 1
            dir = -1
        rnd = self.np_random.random()
        if rnd < self.spawn_probs_cdf[FISH]:
            id = FISH
        elif rnd < self.spawn_probs_cdf[SUBMARINE]:
            id = SUBMARINE
        else:
            id = DIVER
        entity[1] = col
        entity[2] = speed
        entity[3] = dir
        entity[4] = id
        entity[5] = 0
        entity[6] = self.spawn_cooldown
        entity[7] = None

    def collision_with_player(self, row, col, action):
        static_collision = (
            [row, col] == [self.player_row, self.player_col] or
            [row, col] == [self.player_row, self.player_col - self.player_dir]
        )
        # Without this check, the player may "step over" an entity and collision won't be detected.
        # No need to check for old direction (back of the player).
        movement_collision = (
            action in [LEFT, RIGHT] and
            [row, col] == [self.player_row_old, self.player_col_old]
        )
        return static_collision or movement_collision

    def collision_with_entity(self, row, col):
        # Used for player bullets
        for entity in self.entities:
            # Divers are not hit by bullets
            if [row, col] == [entity[0], entity[1]] and entity[4] != DIVER:
                self.despawn(entity)
                return True
        return False

    def _step(self, action: int):
        reward = 0.0
        terminated = False

        # Cooldown
        if self.shoot_timer > 0:
            self.shoot_timer -= 1

        # Deplete oxygen
        self.oxygen_counter += 1
        if self.oxygen_counter == self.oxygen_decay:
            self.oxygen_counter = 0
            self.oxygen -= 1
        if self.oxygen <= 0:
            terminated = True
            self.level_one()
            self._reset()

        # Move player bullet
        for i in range(len(self.player_bullets) - 1, -1, -1):
            row, col, dir = self.player_bullets[i]
            for step in range(2):  # Player bullet moves by 2 tiles per timestep
                col += dir
                if not 0 <= col < self.n_cols or self.collision_with_entity(row, col):
                    self.player_bullets.pop(i)
                    break
                self.player_bullets[i][1] = col

        # Shoot or move
        if action == SHOOT:
            self.shoot()
        else:
            self.player_row_old, self.player_col_old = self.player_row, self.player_col
            self.move(action)

        # Difficulty increases every time the player emerges.
        # Once out, the player can stay at the surface as long as it wants.
        # But as soon as it submerges again, it must collect at least one diver
        # before emerging again, or it will be game over.
        if self.player_row == 0:
            self.oxygen = self.oxygen_max
            self.oxygen_counter = 0
            if self.player_row_old != 0:
                if self.divers_carried == 0:  # Game over
                    terminated = True
                    self.level_one()
                    self._reset()
                else:  # Level up
                    self.level_up()
                    if self.divers_carried == self.divers_carried_max:  # Big reward
                        self.divers_carried = 0
                        reward += self.oxygen
                    else:
                        self.divers_carried -= 1

        # Move entities
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown, b_col = entity

            # Check if out of bounds, and if so check if it's time to respawn
            if col is None:
                cooldown -= 1
                entity[6] = cooldown
                if cooldown > 0:
                    continue
                else:
                    self.respawn(entity)
                    continue

            # Submarines always shoot if they haven't (no cooldown).
            # When they shoot, they don't move.
            if b_col is None and id == SUBMARINE and 0 <= col + dir < self.n_cols:
                entity[7] = col + dir
                if self.collision_with_player(row, b_col, action):
                    terminated = True
                    self.level_one()
                    self._reset()
                    break
                continue

            # Move bullets (one tile faster than its submarine, and never at negative speed)
            if b_col is not None:
                for step in range(max(speed, 0) + 1):
                    b_col += dir
                    if not 0 <= b_col < self.n_cols:
                        entity[7] = None
                        break
                    entity[7] = b_col
                    if self.collision_with_player(row, b_col, action):
                        terminated = True
                        self.level_one()
                        self._reset()
                        break

            # If the speed is negative, check if the entity has waited enough before moving it
            if speed <= 0:
                if timer != speed:
                    entity[5] -= 1
                    # Check if the player moved on an entity that is not moving
                    if self.collision_with_player(row, col, action):
                        if id == DIVER:
                            # Divers are collected if the player has enough room
                            if self.divers_carried < self.divers_carried_max:
                                self.despawn(entity)
                                self.divers_carried += 1
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
            stop_moving = False
            for step in range(speed):
                col += dir
                entity[1] = col
                if not 0 <= col < self.n_cols:
                    self.despawn(entity)
                    break
                if self.collision_with_player(row, col, action):
                    if id == DIVER:
                        if self.divers_carried < self.divers_carried_max:
                            self.despawn(entity)
                            self.divers_carried += 1
                            break
                    else:
                        terminated = True
                        self.level_one()
                        self._reset()
                        break
                for i in range(len(self.player_bullets) - 1, -1, -1):
                    if (
                        id != DIVER and
                        [self.player_bullets[i][0], self.player_bullets[i][1]] == [row, col]
                    ):
                        self.player_bullets.pop(i)
                        self.despawn(entity)
                        stop_moving = True
                        break
                if stop_moving:
                    break

        return self.get_state(), reward, terminated, False, {}

    def _render_board(self):
        import pygame

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw surface
        rect = pygame.Rect((0, 0), (self.window_size[0], self.tile_size[1]))
        pygame.draw.rect(self.window_surface, GRAY, rect)

        # Draw oxygen gauge
        percentage_full = self.oxygen / self.oxygen_max
        rect = pygame.Rect(
            (0, (self.n_rows - 1) * self.tile_size[1]),
            ((self.window_size[0] // 2) * percentage_full, self.tile_size[1]),
        )
        pygame.draw.rect(self.window_surface, PALE_YELLOW, rect)

        # Draw divers gauge
        percentage_full = self.divers_carried / self.divers_carried_max
        rect = pygame.Rect(
            (self.window_size[0] // (1.0 + percentage_full), (self.n_rows - 1) * self.tile_size[1]),
            (self.window_size[0], self.tile_size[1]),
        )
        pygame.draw.rect(self.window_surface, PALE_CYAN, rect)

        # Draw entities and their trail
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown, b_col = entity

            if b_col is not None:
                self.draw_tile(row, b_col, YELLOW)

            if col is None:
                continue

            if id == DIVER:
                color_main = BLUE
                color_trail = CYAN
            elif id == SUBMARINE:
                color_main = RED
                color_trail = PALE_RED
            else:
                color_main = PURPLE
                color_trail = PALE_PURPLE

            self.draw_tile(row, col, color_main)
            speed_scaling = self.speed_chunks[max(timer - speed, 0)]
            for step in range(max(1, speed)):
                col -= dir
                if not 0 <= col < self.n_cols:
                    break
                self.draw_tile(row, col, color_trail, speed_scaling)

        # Draw player bullet
        for i in range(len(self.player_bullets)):
            row, col, dir = self.player_bullets[i]
            self.draw_tile(row, col, WHITE)

        # Draw player
        self.draw_tile(self.player_row, self.player_col, GREEN)
        self.draw_tile(self.player_row, self.player_col - self.player_dir, PALE_GREEN)
