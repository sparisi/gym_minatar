import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional


# action IDs
NOP = 0
LEFT = 1
RIGHT = 2

# state IDs
EMPTY = 0
PADDLE = 1
BALL = 2
TRAIL = 3
BRICK = 4

# color IDs
RED = (255, 0, 0)
PINK = (255, 155, 155)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)


class Breakout(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        size: tuple = (8, 8),
        brick_rows: int = 3,
        levels: int = 3,
        immortal: bool = False,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self.n_rows, self.n_cols = size
        self.brick_rows = brick_rows
        self.immortal = immortal

        # one empty row before and after the bricks, one row for the paddle
        assert (
            self.brick_rows + 3 < self.n_rows
        ), f"cannot fit {brick_rows} brick rows in a board with {self.n_rows} rows"

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            0, 4, (self.n_rows, self.n_cols), dtype=np.int64,
        )

        self.bricks = None
        self.paddle_pos = None
        self.ball_pos = None
        self.last_ball_pos = []
        self.ball_dir = None
        self.last_action = None

        # These two variable control the ball speed: when ball_timesteps == ball_delay,
        # the ball moves. A delay of 3 means that the paddle moves x4 times
        # faster than the ball.
        # When the difficulty increases, ball_delay decreases and can become negative.
        # Negative delay means that the ball is now faster than the paddle.
        self.ball_delay_levels = np.arange(levels - 1, -1, -1) - levels // 2
        self.level = 0
        self.ball_delay = self.ball_delay_levels[self.level]
        self.ball_timesteps = self.ball_delay

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

    def set_state(self, state):
        self.bricks[:] = 0
        self.bricks[state == BRICK] = BRICK
        self.agent_pos = tuple(np.argwhere(state == PADDLE).flatten())
        self.ball_pos = tuple(np.argwhere(state == BALL).flatten())
        self.last_ball_pos = [tuple(pos) for pos in np.argwhere(state == TRAIL)]
        if len(self.ball_pos) == 0:
            self.ball_pos = None
        if len(self.last_ball_pos) == 0:
            self.last_ball_pos = []

    def get_state(self):
        state = np.copy(self.bricks)
        state[self.paddle_pos] = PADDLE
        for pos in self.last_ball_pos:
            state[pos] = TRAIL
        if self.ball_pos is not None:
            state[self.ball_pos] = BALL  # this may overwrite last_ball_pos
        return state

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.paddle_pos = (
            self.n_rows - 1,
            self.np_random.integers((self.n_cols)),
        )
        self.ball_pos = (
            self.np_random.integers(self.brick_rows + 1, self.n_rows - 1),
            self.np_random.integers(self.n_cols),
        )  # do not spawn in paddle or brick rows
        self.ball_dir = (
            -1,
            self.np_random.choice((-1, 1)),
        )  # always spawn going up
        self.ball_delay = self.ball_delay_levels[self.level]
        self.ball_timesteps = self.ball_delay
        self.bricks = np.full((self.n_rows, self.n_cols), EMPTY)
        self.bricks[1 : self.brick_rows + 1, :] = BRICK
        self.last_action = None
        self.last_ball_pos = []

        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _step(self, action: int):
        self.last_action = action
        terminated = False
        reward = 0.0

        if action == NOP:
            pass
        elif action == LEFT:
            self.paddle_pos = (
                self.n_rows - 1,
                max(self.paddle_pos[1] - 1, 0),
            )
        elif action == RIGHT:
            self.paddle_pos = (
                self.n_rows - 1,
                min(self.paddle_pos[1] + 1, self.n_cols - 1),
            )
        else:
            raise ValueError("illegal action")

        if self.ball_delay > 0:
            if self.ball_timesteps != self.ball_delay:
                self.ball_timesteps += 1
                return self.get_state(), reward, terminated, False, {}
            else:
                self.ball_timesteps = 0
                ball_steps = 1
        else:
            ball_steps = abs(self.ball_delay) + 1

        def mid_ball_pos(position, direction):
            front_pos = (  # vertical
                position[0] + direction[0],
                position[1],
            )
            diag_pos = (
                position[0] + direction[0],
                position[1] + direction[1],
            )
            side_pos = (  # horizontal
                position[0],
                position[1] + direction[1],
            )
            return front_pos, diag_pos, side_pos

        self.last_ball_pos = []
        for steps in range(ball_steps):
            self.last_ball_pos.append(self.ball_pos)
            new_ball_pos = (
                self.ball_pos[0] + self.ball_dir[0],
                self.ball_pos[1] + self.ball_dir[1],
            )

            # Collision with bricks cannot happen when collision with ceiling or
            # floor happen
            check_for_bricks = True

            # Collision with side walls
            if new_ball_pos[1] < 0 or new_ball_pos[1] >= self.n_cols:
                new_ball_pos = (
                    new_ball_pos[0],
                    min(max(new_ball_pos[1], 0), self.n_cols - 1),
                )
                self.ball_dir = (
                    self.ball_dir[0],
                    self.ball_dir[1] * (-1),
                )  # bounce

            # Collision with ceiling
            if new_ball_pos[0] < 0:
                new_ball_pos = (0, new_ball_pos[1])
                self.ball_dir = (
                    self.ball_dir[0] * (-1),
                    self.ball_dir[1],
                )
                check_for_bricks = False

            # Collision with floor or paddle
            elif new_ball_pos[0] == self.n_rows - 1:
                check_for_bricks = False
                game_over = True
                front_pos, diag_pos, side_pos = mid_ball_pos(self.ball_pos, self.ball_dir)
                if front_pos == self.paddle_pos:  # keep side direction and bounce up
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0] * (-1),
                        self.ball_dir[1],
                    )
                    game_over = False
                elif diag_pos == self.paddle_pos:  # bounce back (diagonally)
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0] * (-1),
                        self.ball_dir[1] * (-1),
                    )
                    game_over = False
                elif side_pos == self.paddle_pos:  # keep down direction and side bounce
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0],
                        self.ball_dir[1] * (-1),
                    )
                    game_over = True  # hitting the paddle from the side does not save the ball
                elif self.immortal:  # paddle missed ball, but if immortal bounce on the floor
                    new_ball_pos = (
                        self.ball_pos[0],
                        new_ball_pos[1],
                    )
                    self.ball_dir = (
                        self.ball_dir[0] * (-1),
                        self.ball_dir[1],
                    )
                    game_over = False
                    reward = -1.0  # penalize for missing

                if game_over:
                    self.level = 0
                    self.reset()
                    return self.get_state(), reward, True, False, {}

            # Collision with brick (must check after wall collision)
            if check_for_bricks:
                front_pos, diag_pos, side_pos = mid_ball_pos(self.ball_pos, self.ball_dir)
                if self.bricks[front_pos] == BRICK:
                    reward = 1.0
                    self.bricks[front_pos] = EMPTY
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0] * -1,
                        self.ball_dir[1],
                    )
                elif self.bricks[diag_pos] == BRICK:
                    reward = 1.0
                    self.bricks[diag_pos] = EMPTY
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0] * -1,
                        self.ball_dir[1] * -1,
                    )
                elif self.bricks[side_pos] == BRICK:
                    reward = 1.0
                    self.bricks[side_pos] = EMPTY
                    new_ball_pos = self.ball_pos
                    self.ball_dir = (
                        self.ball_dir[0],
                        self.ball_dir[1] * -1,
                    )

            self.ball_pos = new_ball_pos
            if self.bricks.sum() == 0:
                self.level += 1
                if self.level == len(self.ball_delay_levels):
                    self.level -= 1
                    terminated = True
                else:
                    self.reset()
                    # terminated = True

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

        for y in range(self.n_rows):
            for x in range(self.n_cols):
                pos = (x * self.tile_size[0], y * self.tile_size[1])
                rect = pygame.Rect(pos, self.tile_size)
                if state[y][x] == BRICK:
                    pygame.draw.rect(self.window_surface, GRAY, rect)
                elif state[y][x] == PADDLE:
                    pygame.draw.rect(self.window_surface, GREEN, rect)
                elif state[y][x] == TRAIL:
                    pygame.draw.rect(self.window_surface, PINK, rect)
                elif state[y][x] == BALL:
                    pygame.draw.rect(self.window_surface, RED, rect)
                elif state[y][x] == EMPTY:
                    pygame.draw.rect(self.window_surface, BLACK, rect)

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
