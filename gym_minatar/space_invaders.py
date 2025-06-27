import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

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


class SpaceInvaders(Game):
    """
    The player controls a spaceship at the bottom of the board that must shoot
    down waves of aliens.
    - The player moves left/right or stays where it is, and can shoot.
    - Aliens move horizontally and descend when they hit the board left/right
      edges.
        - Their speed increases as they descend: +1 for every time they
          move down for as many times as the initial number of alien rows.
          For example, if at the beginning there are 3 alien rows, the speed
          increases by 1 after they reach row 6, then 9, ... and so on.
        - Every level increases the speed by as many frames as the number of
          initial alien rows.
        - At their fastest, the aliens move as fast as the player.
    - Aliens shoot whenever possible and act as one entity: once their cooldown is
      over, a random alien shoots.
        - Alien bullets move as fast as the player, regardless of the aliens' speed.
    - The player also has to wait to cooldown before shooting again.
    - The player receives +1 for hitting an alien with its bullets.
    - The game ends if the aliens reach the bottom, or if the player is hit by
      their bullets.
    - If the player destroys all aliens, the next level starts.
    - Difficulty increases with levels, making aliens start closer to the player.
    - The observation space is a 4-channel grid with 0s for empty tiles, and 1 or -1
      for information about the game entities:
        - Channel 0: player position (1).
        - Channel 1: aliens (-1 moving left, 1 moving right).
        - Channel 2: player bullets (-1 moving up).
        - Channel 3: alien bullets (1 moving down).
    """

    def __init__(self, aliens_rows: int = 3, **kwargs):
        Game.__init__(self, **kwargs)

        self.aliens_rows = aliens_rows

        assert (
            self.aliens_rows > 0
        ), f"aliens rows must be positive (received {self.n_rows})"
        # First two and last two columns must be empty at the beginning
        assert self.n_cols > 4, f"board too small ({self.n_cols} columns, minimum is 5)"
        # One row for the player, one for the aliens, one to allow aliens to move down once
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows, minimum is 3)"
        # One empty row below the aliens, one row for the player
        assert (
            self.n_rows >= self.aliens_rows + 2
        ), f"cannot fit {aliens_rows} alien rows in a board with {self.n_rows} rows"

        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 4), dtype=np.int64,
        )  # fmt: skip
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
            "shoot": 3,
        }

        self.state = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        self.aliens_dir = None
        self.bottom_alien = None  # Track lowest row for faster rendering
        self.aliens_move_down = None
        self.player_pos = None
        self.last_action = None
        self.player_shoot_cooldown = 3
        self.player_shoot_timer = 0
        self.alien_shoot_cooldown = 5
        self.alien_shoot_timer = 0
        self.starting_row = 0  # Denotes the current level of the game

        # These two variables control the aliens speed: when aliens_timesteps == aliens_delay,
        # the aliens move. A delay of 1 means that aliens need 2 timestep to
        # move. A delay of 0 means that they move as fast as the player (1 tile
        # per timestep).
        # This is very similar to what other games do, with the difference that
        # we don't need traces to encode speed in observations, because speed
        # depends only on how far the aliens have descended.
        self.aliens_delay_levels = np.arange(self.n_rows // self.aliens_rows - 1, -1, -1)
        self.aliens_delay_levels *= self.aliens_rows
        self.aliens_delay = self.aliens_delay_levels[0]
        self.aliens_timesteps = 0
        self.lowest_row_reached = self.bottom_alien

    def get_state(self):
        return self.state.copy()

    def level_one(self):
        self.starting_row = 0
        self._reset()

    def level_up(self):
        self.starting_row = min(self.starting_row + 1, self.n_rows - self.aliens_rows - 1)
        self._reset()

    def _reset(self, seed: int = None, **kwargs):
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
        self.state[
            self.starting_row : self.starting_row + self.aliens_rows, :, 1
        ] = self.aliens_dir
        self.state[:, 0, 1] = 0  # First two and last two columns are empty
        self.state[:, 1, 1] = 0
        self.state[:, -2, 1] = 0
        self.state[:, -1, 1] = 0
        self.bottom_alien = self.starting_row + self.aliens_rows - 1
        self.lowest_row_reached = self.bottom_alien
        self.aliens_move_down = False

        return self.get_state(), {}

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
        # Row is the same as alien's because the bullet will be moved down in the same turn
        self.state[who_shoots[0], who_shoots[1], 3] = 1

    def _step(self, action: int):
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

        def move_bullets():  # Also check if player bullets hit aliens
            self.state[..., 2] = np.roll(self.state[..., 2], -1, 0)  # Player bullets up
            self.state[-1, :, 2] = 0  # Remove if rolled back to bottom
            self.state[..., 3] = np.roll(self.state[..., 3], 1, 0)  # Alien bullets down
            self.state[0, :, 3] = 0  # Remove if rolled back to top
            alien_hits = self.state[..., 2] * self.state[..., 1] * self.aliens_dir * -1
            self.state[..., 1] *= 1 - alien_hits
            self.state[..., 2] *= 1 - alien_hits

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
                self.lowest_row_reached = max(
                    self.bottom_alien, self.lowest_row_reached
                )
                self.aliens_move_down = False
                delay_level = (self.lowest_row_reached + self.aliens_rows) // self.aliens_rows - 2
                delay_level = min(delay_level, len(self.aliens_delay_levels) - 1)  # To prevent an error when aliens reach the bottom row
                self.aliens_delay = self.aliens_delay_levels[delay_level]
            else:
                self.state[..., 1] = np.roll(self.state[..., 1], self.aliens_dir, 1)
                if (  # Aliens hit the edge of the board
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

    def _render_board(self):
        import pygame

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw aliens
        color = RED if self.aliens_dir == 1 or self.no_trail else PALE_RED
        for x in range(self.bottom_alien - self.aliens_rows + 1, self.bottom_alien + 1):
            for y in range(self.n_cols):
                if self.state[x, y, 1]:
                    self.draw_tile(x, y, color)

        # for x in range(self.bottom_alien - self.aliens_rows + 1, self.n_rows):
        for x in range(self.n_rows):
            for y in range(self.n_cols):
                if self.state[x, y, 3] == 1:
                    self.draw_tile(x, y, YELLOW)
                elif self.state[x, y, 2] == -1:
                    self.draw_tile(x, y, WHITE)

        # Draw player
        self.draw_tile(self.player_pos[0], self.player_pos[1], GREEN)
