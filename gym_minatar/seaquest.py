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

BLACK = (0, 0, 0)
GRAY = (100, 100, 100)  # surface
RED = (255, 0, 0)  # submarines
PINK = (255, 155, 155)  # submarines trail
YELLOW = (255, 255, 0)  # submarines bullet
PURPLE = (102, 51, 153)
PALE_PURPLE = (200, 155, 255)  # fish trail
BLUE = (0, 0, 255)  # divers
CYAN = (0, 255, 255)  # divers trail
PALE_CYAN = (200, 255, 255)  # divers bar
GREEN = (0, 255, 0)  # player front
DARK_GREEN = (0, 155, 0)  # player back
PALE_GREEN = (200, 255, 200)  # player oxygen
WHITE = (255, 255, 255)  # player bullet


class Seaquest(gym.Env):
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
        self.n_rows += 2  # 2 bottom rows are for oxygen and divers carried

        self.player_dir = None
        self.player_row = None
        self.player_col = None
        self.cooldowns = None

        # Lists of [row, col, direction, bullet_col] (bullets only for submarines)
        self.bullets = []  # shot by the player
        self.fishes = []  # do not shoot
        self.submarines = []  # shoot
        self.divers = []  # can be picked

        self.enemy_speed = 1  # fish and submarines
        self.enemy_speed_start = 1
        self.diver_speed = 1

        self.oxygen_max = 10
        self.oxygen_decay = 10  # number of steps before oxygen goes down by 1
        self.shoot_cooldown_max = 5
        self.divers_carried_max = 6
        self.spawn_cooldown = 5

        self.oxygen = 0
        self.oxygen_counter = 0
        self.shoot_cooldown = 0
        self.divers_carried = 0
        self.cooldowns = []

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            0, 99,
            (self.n_rows, self.n_cols),
            dtype=np.int64,
        )

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (
            min(64 * self.n_cols, 512),
            min(64 * self.n_rows, 512),
        )
        self.tile_size = (
            self.window_size[0] // self.n_cols,
            self.window_size[1] // self.n_rows,
        )

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.last_action = None

        self.player_row = self.n_rows - 2  # last row is for oxygen and divers carried
        self.player_dir = 1 if self.np_random.random() < 0.5 else -1
        if self.player_dir == 1:  # facing right
            self.player_col = self.np_random.integers(self.n_cols - 1)
        else:  # facing left
            self.player_col = self.np_random.integers(1, self.n_cols)

        # Start with empty board and random spawns (last row is for oxygen and divers carried, first row for surface)
        self.cooldowns = self.np_random.integers(0, self.spawn_cooldown, self.n_rows - 2)
        self.bullets = []  # shot by the player
        self.fishes = []  # do not shoot
        self.submarines = []  # shoot
        self.divers = []  # can be picked
        self.shoot_cooldown = 0
        self.oxygen = self.oxygen_max
        self.oxygen_counter = 0
        self.divers_carried = 0
        self.enemy_speed = self.enemy_speed_start

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def move(self, a):
        if a == LEFT:
            self.player_col = max(self.player_col - 1, 0)
            if self.player_dir == 1:  # turn
                self.player_dir = -1
        elif a == DOWN:
            self.player_row = min(self.player_row + 1, self.n_rows - 2)  # last row is for oxygen and divers carried
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

    def spawn(self, row):
        if self.np_random.random() < 0.5:
            col = 0
            direction = 1
        else:
            col = self.n_cols - 1
            direction = -1
        if self.np_random.random() < 1.0 / 3.0:
            self.fishes.append([row, col, direction])
        elif self.np_random.random() < 2.0 / 3.0:
            self.submarines.append([row, col, direction, None])
        else:
            self.divers.append([row, col, direction])
        self.cooldowns[row - 1] = self.n_cols + self.spawn_cooldown + 1  # first row is for surface

    def collision_player(self, row, col):
        return (
            [row, col] == [self.player_row, self.player_col] or
            [row, col] == [self.player_row, self.player_col + 1]
        )

    def collision_bullet(self, row, col):
        for i in range(len(self.bullets)):
            b_row, b_col, b_dir = self.bullets[i]
            if [row, col] == [b_row, b_col]:
                return i
        return None

    def ramp_difficulty(self):
        return

    def shooting(self, action):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.shoot_cooldown > 0 or action != SHOOT:
            return
        b_col = self.player_col + self.player_dir
        if not (0 <= b_col < self.n_rows):  # out of bound bullet
            return
        self.shoot_cooldown = self.shoot_cooldown_max
        self.bullets.append([self.player_row, b_col, self.player_dir])

    def get_state(self):
        return np.zeros((self.n_rows, self.n_cols))

    def step(self, action: int):
        if self.player_row == -1:  # game ended but not reset yet
            return self.get_state(), -1.0, True, False, {}

        reward = 0.0
        terminated = False
        self.last_action = action

        # Deplete oxygen
        self.oxygen_counter += 1
        if self.oxygen_counter == self.oxygen_decay:
            self.oxygen_counter = 0
            self.oxygen -= 1
        if self.oxygen < 0:
            terminated = True

        # Move player bullet (note: they don't collide with submarines bullets)
        for i in range(len(self.bullets) - 1, -1, -1):
            row, col, dir = self.bullets[i]
            stop_moving = False
            for j in range(2):  # player moves at speed 1, its bullet at twice speed
                col += dir
                self.bullets[i][1] = col
                if not (0 <= col < self.n_cols):  # out of bounds
                    self.bullets.pop(i)
                    break
                for k in range(len(self.fishes) -1, -1, -1):
                    enemy_row, enemy_col, enemy_dir = self.fishes[k]
                    if [enemy_row, enemy_col] == [row, col]:
                        self.fishes.pop(k)
                        self.bullets.pop(i)
                        stop_moving = True
                        break
                for k in range(len(self.submarines) - 1, -1, -1):
                    enemy_row, enemy_col, enemy_dir, enemy_bullet_col = self.submarines[k]
                    if [enemy_row, enemy_col] == [row, col]:
                        self.submarines.pop(k)
                        self.bullets.pop(i)
                        stop_moving = True
                        break
                    elif [enemy_bullet_col, enemy_col] == [row, col]:
                        # self.submarines[k][3] = None  # remove submarine bullet
                        self.bullets.pop(i)
                        stop_moving = True
                        break
                if stop_moving:
                    break

        # Move player or shoot
        old_player_row = self.player_row
        self.shooting(action)
        self.move(action)

        # At the surface
        if self.player_row == 0:
            if self.divers_carried == 0:
                if old_player_row == 0:  # once out, the player can stay at the surface as long as it wants
                    self.oxygen = self.oxygen_max
                    self.oxygen_counter = 0
                else:  # but if submerged, it cannot re-emerge without having collected at least one diver
                    terminated = True
            else:
                if self.divers_carried == self.divers_carried_max:
                    reward += self.oxygen
                    self.divers_carried = 0
                else:
                    self.divers_carried -= 1
                self.oxygen = self.oxygen_max
                self.oxygen_counter = 0
                self.ramp_difficulty()

        # Move fishes
        for i in range(len(self.fishes) - 1, -1, -1):
            row, col, direction = self.fishes[i]
            for j in range(self.enemy_speed):
                col += direction
                self.fishes[i][1] = col
                if not (0 <= col < self.n_cols):  # out of bounds
                    self.fishes.pop(i)
                    self.cooldowns[row - 1] = self.spawn_cooldown
                    break
                if self.collision_player(row, col):
                    terminated = True
                bullet = self.collision_bullet(row, col)
                if bullet is not None:
                    reward += 1.0
                    self.bullets.pop(bullet)
                    self.fishes.pop(i)

        # Move submarines and their bullets
        for i in range(len(self.submarines) - 1, -1, -1):
            row, col, direction, bullet_col = self.submarines[i]
            if bullet_col is None:  # always shoot, unless their bullet is still on the board
                self.submarines[i][3] = col + direction
            else:
                for j in range(self.enemy_speed):
                    col += direction
                    self.submarines[i][1] = col
                    bullet_col += direction + self.enemy_speed * direction  # bullets move twice as fast
                    self.submarines[i][3] = bullet_col
                    if not (0 <= bullet_col < self.n_cols):  # out of bounds
                        self.submarines[i][3] = None
                    if not (0 <= col < self.n_cols):  # out of bounds
                        self.submarines.pop(i)
                        self.cooldowns[row - 1] = self.spawn_cooldown
                        break
                    if self.collision_player(row, col) or self.collision_player(row, bullet_col):
                        terminated = True
                    bullet = self.collision_bullet(row, col)  # collision with player bullet
                    if bullet is not None:
                        reward += 1.0
                        self.bullets.pop(bullet)
                        self.submarines.pop(i)

        # Move divers
        for i in range(len(self.divers) - 1, -1, -1):
            row, col, direction = self.divers[i]
            for j in range(self.diver_speed):
                col += direction
                self.divers[i][1] = col
                if not (0 <= col < self.n_cols):  # out of bounds
                    self.divers.pop(i)
                    self.cooldowns[row - 1] = self.spawn_cooldown
                    break
                if self.collision_player(row, col):
                    if self.divers_carried < self.divers_carried_max:
                        self.divers_carried += 1
                        self.divers.pop(i)

        # Spawn enemies and divers
        for i in range(len(self.cooldowns)):
            if self.cooldowns[i] > 0:  # still cooling down
                self.cooldowns[i] -= 1
            else:
                self.spawn(i + 1)

        if terminated:
            self.player_row = -1

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
                pygame.display.set_caption("Seaquest")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None, "Something went wrong with pygame."

        if self.clock is None:
            self.clock = pygame.time.Clock()

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
        pygame.draw.rect(self.window_surface, PALE_GREEN, rect)

        # Draw divers bar
        percentage_full = self.divers_carried / self.divers_carried_max
        rect = pygame.Rect(
            (self.window_size[0] // (1.0 + percentage_full), (self.n_rows - 1) * self.tile_size[1]),
            (self.window_size[0], self.tile_size[1]),
        )
        pygame.draw.rect(self.window_surface, PALE_CYAN, rect)

        def draw_tile(row, col, color):
            pos = (col * self.tile_size[0], row * self.tile_size[1])
            rect = pygame.Rect(pos, self.tile_size)
            pygame.draw.rect(self.window_surface, color, rect)

        # Draw fishes
        for i in range(len(self.fishes)):
            row, col, direction = self.fishes[i]
            draw_tile(row, col, PURPLE)
            draw_tile(row, col - direction, PALE_PURPLE)

        # Draw submarines
        for i in range(len(self.submarines)):
            row, col, direction, b_col = self.submarines[i]
            draw_tile(row, col, RED)
            draw_tile(row, col - direction, PINK)
            if b_col is not None:
                draw_tile(row, b_col, YELLOW)

        # Draw divers
        for i in range(len(self.divers)):
            row, col, direction = self.divers[i]
            draw_tile(row, col, BLUE)
            draw_tile(row, col - direction, CYAN)

        # Draw player's bullet
        for i in range(len(self.bullets)):
            row, col, direction = self.bullets[i]
            draw_tile(row, col, WHITE)

        # Draw player
        draw_tile(self.player_row, self.player_col, GREEN)
        draw_tile(self.player_row, self.player_col - self.player_dir, DARK_GREEN)

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
