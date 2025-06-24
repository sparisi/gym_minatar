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
PALE_CYAN = (200, 255, 255)  # diver bar
GREEN = (0, 255, 0)  # player front
PALE_GREEN = (200, 255, 200)  # player back
WHITE = (255, 255, 255)  # player bullet
# PALE_GRAY = (235, 235, 235)  # player bullet trail
PALE_YELLOW = (255, 255, 200)  # player oxygen

# Bullets don't have a trail because their direction can be inferred from the
# submarine / player position. Bullets travel faster than the entity that shot them,
# so a bullet to the left of the agent can only be going left.

class Seaquest(gym.Env):
    """
    The player controls a submarine that can move and shoot, collecting divers
    and avoiding enemies.
    - The player occupies two horizontal tiles and can move left/right/up/down,
      or not move at all.
        - If the player is facing left (or right) and moves right (or left),
          it turns around (front and back swap).
    - The player can shoot in front of itself.
        - Shooting has a cooldown.
        - The bullet moves twice as fast as the player.
        - Player bullets destroy enemies on collision, giving a reward to the player.
    - Enemies and divers move at random speeds and leave a trail behind them
      to infer their direction.
    - Enemies can be fishes or submarines.
        - Submarines shoot, fishes don't.
        - As soon as their bullet leaves the screen, they shoot a new one.
        - Their bullet moves 1 frames faster than them.
        - Player and enemies bullets don't collide.
        - If the player is hit by the enemies bullets, it dies.
    - The player can collect divers (up to 6) on collision.
    - The player has an oxygen reserve (bar at the bottom left) that depletes
      over time. When it's empty, the game ends.
    - The player can emerge by moving to the top.
      - If the player is carrying 6 divers, it gets a reward proportional to the
        amount of oxygen left.
      - If the player is carrying at least 1 divers, its oxygen is fully restored.
      - If the player is not carrying any divers, the game ends.
      - The player can stay at the surface as long as it wants without depleting
        oxygen.
    - When the player submerges again, the difficulty level increases
      (enemies and divers move faster, respawn time decreases).
    - The observation space is a 3-channel grid with 0s for empty tiles, and 1 or -1
      for information about the game entities:
        - Channel 0: oxygen and diver bars (denoted by 1s at the bottom of the grid),
          player position and bullets (-1 moving left, 1 moving right).
        - Channel 1: fishes and their trails (-1 moving left, 1 moving right).
        - Channel 2: submarines, bullets, and their trails (-1 moving left, 1 moving right).
        - Channel 3: divers and their trails (-1 moving left, 1 moving right).
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
        # Last row is for oxygen and divers bar, first row is the surface
        self.n_rows, self.n_cols = size
        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        self.n_rows += 2

        self.n_rows, self.n_cols = size
        self.spawn_cooldown = 10
        self.max_entity_speed = -2
        self.entities = None
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

        # First channel for oxygen and diver bars, player and its bullet.
        # Second channel for fishes and their trail.
        # Third channel for submarines and their trail.
        # Fourth channel for divers and their trail.
        # For moving entities, -1 means movement to the left, +1 to the right.
        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 4),
        )
        self.action_space = gym.spaces.Discrete(6)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
            "shoot": 5,
        }

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
        )

    def get_state(self):
        state = np.zeros(self.observation_space.shape)
        state[self.player_row, self.player_col, 0] = self.player_dir
        state[self.player_row, self.player_col - self.player_dir, 0] = self.player_dir
        for bullet in self.player_bullets:
            row, col, dir = bullet
            state[row, col, 0] = dir
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown, b_col = entity
            if col is None:
                continue
            if speed <= 0:
                if timer != speed:
                    speed_scaling = (timer - 0.5) / speed
                    state[row, col, 1] = dir
                    if 0 <= col - dir < self.n_cols:
                        state[row, (col - dir), id] = dir * speed_scaling
                    continue
                else:
                    speed = 1
            for step in range(speed + 1):
                if not 0 <= col - step * dir < self.n_cols:
                    break
                state[row, col - step * dir, id] = dir
            if b_col is not None and 0 <= b_col < self.n_cols:
                state[row, b_col, id] = dir
        return state

    def level_one(self):
        self.max_entity_speed = -2
        self.spawn_cooldown = 10

    def level_up(self):
        self.max_entity_speed = min(self.max_entity_speed + 1, self.n_rows - 1)
        self.spawn_cooldown = max(self.spawn_cooldown - 1, 0)

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_action = None

        self.shoot_timer = 0
        self.player_bullets = []  # Dynamic list because there can be many bullets on the board
        self.player_row = self.n_rows - 2
        self.player_dir = 1 if self.np_random.random() < 0.5 else -1
        if self.player_dir == 1:  # Facing right
            self.player_col = self.np_random.integers(1, self.n_cols)
        else:  # Facing left
            self.player_col = self.np_random.integers(0, self.n_cols - 1)
        self.player_row_old, self.player_col_old = self.player_row, self.player_col

        # Entries are denoted by (row, col, speed, direction, id, timer, cooldown, bullet_col).
        # Timer is for entries with negative speed (they move slower than the player).
        # Cooldown is for respawing.
        # Bullet column is None except for submarines.
        # First and last row of the board are empty.
        # Board is empty at the beginning (col is None) with random cooldowns,
        # so entries will spawn little by little.
        speeds = self.np_random.integers(self.max_entity_speed - 2, self.max_entity_speed + 1, self.n_rows - 2)
        dirs = np.sign(self.np_random.uniform(-1, 1, self.n_rows - 2)).astype(np.int64)
        rows = np.arange(1, self.n_rows - 1)
        ids = self.np_random.random(self.n_rows - 2)
        ids = (ids[None] < np.array([0.25, 0.5, 1.0])[..., None]).sum(0)  # 50% fish, 25% submarines, 25% divers
        cdowns = self.np_random.integers(0, self.spawn_cooldown, self.n_rows - 2)
        self.entities = [[r, None, s, d, i, 0, cd, None] for r, s, d, i, cd in zip(rows, speeds, dirs, ids, cdowns)]

        self.oxygen = self.oxygen_max
        self.oxygen_counter = 0
        self.divers_carried = 0

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def shoot(self):
        if self.shoot_timer > 0:
            return
        self.shoot_timer = self.shoot_cooldown
        col = self.player_col + self.player_dir
        if not (0 <= col < self.n_cols):
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
        speed = self.np_random.integers(self.max_entity_speed - 2, self.max_entity_speed + 1)
        if self.np_random.random() < 0.5:
            col = 0
            dir = 1
        else:
            col = self.n_cols - 1
            dir = -1
        id = self.np_random.random()
        if id < 0.5:
            id = FISH
        elif id < 0.75:
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
        # Must check old position, otherwise the player may "step over"
        # an entity and collision won't be detected.
        # No need to check for old direction (back of the player).
        return (
            ([row, col] == [self.player_row, self.player_col]) or
            ([row, col] == [self.player_row, self.player_col - self.player_dir]) or
            (action in [LEFT, RIGHT] and ([row, col] == [self.player_row_old, self.player_col_old]))
        )

    def collision_with_entity(self, row, col):
        # Used for player bullets
        for entity in self.entities:
            # Divers are not hit by bullets
            if [row, col] == [entity[0], entity[1]] and entity[4] != DIVER:
                self.despawn(entity)
                return True
        return False

    def step(self, action: int):
        reward = 0.0
        terminated = False
        self.last_action = action

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
            self.reset()

        # Move player bullet
        for i in range(len(self.player_bullets) - 1, -1, -1):
            row, col, dir = self.player_bullets[i]
            for step in range(2):  # Player bullet moves by 2 tiles
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
        # or it will be game over.
        if self.player_row == 0:
            self.oxygen = self.oxygen_max
            self.oxygen_counter = 0
            if self.player_row_old != 0:
                if self.divers_carried == 0:  # Game over
                    terminated = True
                    self.level_one()
                    self.reset()
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
            if col == None:
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
                    self.reset()
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
                        self.reset()
                        break

            # If the speed is negative, check if the entity has waited enough before moving it
            if speed <= 0:
                if timer != speed:
                    entity[5] -= 1
                    # Check if the player moved on an entity that is not moving
                    if self.collision_with_player(row, col, action):
                        if id == DIVER:  # Divers are collected if the player has enough room
                            if self.divers_carried < self.divers_carried_max:
                                self.despawn(entity)
                                self.divers_carried += 1
                                break
                        else:
                            terminated = True
                            self.level_one()
                            self.reset()
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
                    if id == DIVER:  # Divers are collected if the player has enough room
                        if self.divers_carried < self.divers_carried_max:
                            self.despawn(entity)
                            self.divers_carried += 1
                            break
                    else:
                        terminated = True
                        self.level_one()
                        self.reset()
                        break
                for i in range(len(self.player_bullets) - 1, -1, -1):
                    if [self.player_bullets[i][0], self.player_bullets[i][1]] == [row, col] and id != DIVER:
                        self.player_bullets.pop(i)
                        self.despawn(entity)
                        stop_moving = True
                        break
                if stop_moving:
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

        state = self.get_state()

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw surface
        rect = pygame.Rect((0, 0), (self.window_size[0], self.tile_size[1]))
        pygame.draw.rect(self.window_surface, GRAY, rect)

        # Draw oxygen bar
        percentage_full = self.oxygen / self.oxygen_max
        rect = pygame.Rect(
            (0, (self.n_rows - 1) * self.tile_size[1]),
            ((self.window_size[0] // 2) * percentage_full, self.tile_size[1]),
        )
        pygame.draw.rect(self.window_surface, PALE_YELLOW, rect)

        # Draw divers bar
        percentage_full = self.divers_carried / self.divers_carried_max
        rect = pygame.Rect(
            (self.window_size[0] // (1.0 + percentage_full), (self.n_rows - 1) * self.tile_size[1]),
            (self.window_size[0], self.tile_size[1]),
        )
        pygame.draw.rect(self.window_surface, PALE_CYAN, rect)

        def draw_tile(row, col, color, scale=1.0):
            pos = (col * self.tile_size[0], row * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            if scale != 1.0:
                rect = rect.scale_by(scale)
            pygame.draw.rect(self.window_surface, color, rect)

        # Draw entities and their trail
        for entity in self.entities:
            row, col, speed, dir, id, timer, cooldown, b_col = entity
            if b_col is not None:
                draw_tile(row, b_col, YELLOW)
            if col == None:
                continue
            if id == DIVER:
                color = BLUE
                color_trail = CYAN
            elif id == SUBMARINE:
                color = RED
                color_trail = PALE_RED
            else:
                color = PURPLE
                color_trail = PALE_PURPLE
            draw_tile(row, col, color)
            if speed <= 0:
                if timer != speed:
                    if 0 <= col < self.n_cols:
                        col = (col - dir)  # Backward for trail
                        speed_scaling = (timer - 0.5) / speed
                        draw_tile(row, col, color_trail, scale=speed_scaling)
                    continue
                else:
                    speed = 1
            for step in range(max(0, speed)):
                col -= dir
                if not 0 <= col < self.n_cols:
                    break
                draw_tile(row, col, color_trail)

        # Draw player bullet
        for i in range(len(self.player_bullets)):
            row, col, dir = self.player_bullets[i]
            draw_tile(row, col, WHITE)

        # Draw player
        draw_tile(self.player_row, self.player_col, GREEN)
        draw_tile(self.player_row, self.player_col - self.player_dir, PALE_GREEN)

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
