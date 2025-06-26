import numpy as np
import gymnasium as gym
from gym_minatar.minatar_game import Game

# Action IDs
NOP = 0
LEFT = 1
RIGHT = 2

BLUE = (0, 0, 255)  # ball
CYAN = (0, 255, 255)  # ball trail
GREEN = (0, 255, 0)  # player
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)  # bricks


class Breakout(Game):
    """
    The player controls a paddle to bounce a ball and break bricks.
    - The paddle moves left/right at the bottom of the grid, or stand still.
    - The ball moves diagonally and bounces off the paddle, walls, and bricks.
    - The player receives +1 for breaking bricks.
    - When all bricks are destroyed, a new game starts at increased difficulty.
      - Difficulty increases ball speed.
    - The game ends if the ball falls below the paddle.
    - The observation space is a 3-channel grid with 0s for empty tiles, and
      values in [-1, 1] for game entities:
        - Channel 0: paddle position (1).
        - Channel 1: bricks (1s).
        - Channel 2: ball and its trail (-1 moving up, 1 moving down).
        - Intermediate values in (-1, 1) denote the ball speed when it moves slower
          than 1 tile per timestep.
    """

    def __init__(self, brick_rows: int = 3, **kwargs):
        Game.__init__(self, **kwargs)

        self.brick_rows = brick_rows

        assert self.n_cols > 2, f"board too small ({self.n_cols} columns)"
        assert self.n_rows > 2, f"board too small ({self.n_rows} rows)"
        assert (  # One empty row before and after the bricks, one row for the paddle
            self.brick_rows + 3 < self.n_rows
        ), f"cannot fit {brick_rows} brick rows in a board with {self.n_rows} rows"

        self.observation_space = gym.spaces.Box(
            -1, 1, (self.n_rows, self.n_cols, 3),
        )  # fmt: skip
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            "nop": 0,
            "left": 1,
            "right": 2,
        }

        self.bricks = np.zeros((self.n_rows, self.n_cols), dtype=np.int64)
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_dir = None
        self.last_action = None
        self.contact_pos = None  # To store ball contact with brick or paddle
        self.last_ball_pos = []  # To store trails

        # Please see freeway.py for more details about these variables
        self.timer = 0
        self.init_speed = -1
        self.max_speed = 1
        self.speed = self.init_speed
        n_slow_speeds = self.init_speed - 1
        self.slow_speed_bins = np.arange(n_slow_speeds, 0) / n_slow_speeds

    def get_state(self):
        state = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        state[..., 1] = self.bricks
        state[self.paddle_pos[0], self.paddle_pos[1], 0] = 1
        state[self.ball_pos[0], self.ball_pos[1], 2] = self.ball_dir[0]
        if self.contact_pos is not None:
            state[self.contact_pos[0], self.contact_pos[1], 2] = self.ball_dir[0]
        speed_scaling = self.slow_speed_bins[max(self.timer - self.speed, 0)]
        for ball_pos in self.last_ball_pos:
            state[ball_pos[0], ball_pos[1], 2] = self.ball_dir[0] * speed_scaling
        return state

    def level_one(self):
        self.speed = self.init_speed

    def level_up(self):
        self.speed = min(self.speed + 1, self.max_speed)

    def _reset(self, seed: int = None, **kwargs):
        self.paddle_pos = [
            self.n_rows - 1,
            self.np_random.integers((self.n_cols)),
        ]
        self.ball_pos = [
            self.np_random.integers(self.brick_rows + 1, self.n_rows - 1),
            self.np_random.integers(self.n_cols),
        ]  # Do not spawn in paddle or brick rows
        self.ball_dir = [
            -1,
            self.np_random.choice((-1, 1)),
        ]  # Always spawn going up
        self.timer = 0
        self.bricks[:] = 0
        self.bricks[1 : self.brick_rows + 1, :] = 1
        self.last_ball_pos = []
        self.contact_pos = None
        return self.get_state(), {}

    def _step(self, action: int):
        self.contact_pos = None
        terminated = False
        reward = 0.0

        # Move paddle
        if action == NOP:
            pass
        elif action == LEFT:
            self.paddle_pos[1] = max(self.paddle_pos[1] - 1, 0)
        elif action == RIGHT:
            self.paddle_pos[1] = min(self.paddle_pos[1] + 1, self.n_cols - 1)
        else:
            raise ValueError("illegal action")

        # Check if it's time to move the ball
        speed = self.speed
        if self.speed < 0:
            if self.timer > self.speed:
                self.timer -= 1
                return self.get_state(), reward, terminated, False, {}
            else:
                self.timer = 0
                speed = 0

        def where_ball_is_going(position, direction):
            front_pos = [  # Vertical
                position[0] + direction[0],
                position[1],
            ]
            diag_pos = [
                position[0] + direction[0],
                position[1] + direction[1],
            ]
            side_pos = [  # Horizontal
                position[0],
                position[1] + direction[1],
            ]
            return front_pos, diag_pos, side_pos

        self.last_ball_pos = []
        for steps in range(speed + 1):
            self.last_ball_pos.append(self.ball_pos)
            new_ball_pos = [
                self.ball_pos[0] + self.ball_dir[0],
                self.ball_pos[1] + self.ball_dir[1],
            ]

            # Collision with bricks can't happen when collision with ceiling or floor happens
            check_for_bricks = True

            # Collision with side walls
            if new_ball_pos[1] < 0 or new_ball_pos[1] >= self.n_cols:
                new_ball_pos[1] = min(max(new_ball_pos[1], 0), self.n_cols - 1)
                self.ball_dir[1] *= -1  # Bounce

            # Collision with ceiling
            if new_ball_pos[0] < 0:
                new_ball_pos[0] = 0
                self.ball_dir[0] *= -1
                check_for_bricks = False

            # Collision with floor or paddle
            elif new_ball_pos[0] == self.n_rows - 1:
                check_for_bricks = False
                game_over = True
                front_pos, diag_pos, side_pos = where_ball_is_going(
                    self.ball_pos, self.ball_dir
                )
                if front_pos == self.paddle_pos:  # Keep side direction and bounce up
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    game_over = False
                    self.contact_pos = front_pos
                elif diag_pos == self.paddle_pos:  # Bounce back (diagonally)
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.ball_dir[1] *= -1
                    game_over = False
                    self.contact_pos = diag_pos
                elif side_pos == self.paddle_pos:  # Keep down direction and side bounce
                    new_ball_pos = self.ball_pos
                    self.ball_dir[1] *= -1
                    game_over = True  # Hitting the ball from the side does not save it
                    self.contact_pos = side_pos
                if game_over:
                    terminated = True
                    self.level_one()
                    self._reset()
                    return self.get_state(), reward, terminated, False, {}

            # Collision with brick (must check after wall collision)
            if check_for_bricks:
                front_pos, diag_pos, side_pos = where_ball_is_going(
                    self.ball_pos, self.ball_dir
                )
                if self.bricks[front_pos[0], front_pos[1]]:
                    reward = 1.0
                    self.bricks[front_pos[0], front_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.contact_pos = front_pos
                elif self.bricks[diag_pos[0], diag_pos[1]]:
                    reward = 1.0
                    self.bricks[diag_pos[0], diag_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[0] *= -1
                    self.ball_dir[1] *= -1
                    self.contact_pos = diag_pos
                elif self.bricks[side_pos[0], side_pos[1]]:
                    reward = 1.0
                    self.bricks[side_pos[0], side_pos[1]] = 0
                    new_ball_pos = self.ball_pos
                    self.ball_dir[1] *= -1
                    self.contact_pos = side_pos

            self.ball_pos = new_ball_pos

            if self.bricks.sum() == 0:
                if self.level == len(self.ball_delay_levels) - 1:
                    terminated = True
                else:
                    self.level_up()
                    self._reset()

        return self.get_state(), reward, terminated, False, {}

    def _render_board(self):
        import pygame

        # Draw background
        rect = pygame.Rect((0, 0), self.window_size)
        pygame.draw.rect(self.window_surface, BLACK, rect)

        # Draw bricks
        for x in range(1, self.brick_rows + 1):
            for y in range(self.n_cols):
                if self.bricks[x, y]:
                    self.draw_tile(x, y, GRAY)

        # Draw trail
        speed_scaling = self.slow_speed_bins[max(self.timer - self.speed, 0)]
        for trail in self.last_ball_pos:
            self.draw_tile(trail[0], trail[1], CYAN, speed_scaling)

        # Draw ball
        self.draw_tile(self.ball_pos[0], self.ball_pos[1], BLUE)

        # Draw trail if there was a collision
        if self.contact_pos is not None:
            self.draw_tile(self.contact_pos[0], self.contact_pos[1], CYAN)

        # Draw paddle
        self.draw_tile(self.paddle_pos[0], self.paddle_pos[1], GREEN)
