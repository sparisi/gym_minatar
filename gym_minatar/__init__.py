from gymnasium.envs.registration import register

register(
    id="Gym-MinAtar/Breakout-v1",
    entry_point="gym_minatar.breakout:Breakout",
    max_episode_steps=10_000,
)

register(
    id="Gym-MinAtar/Freeway-v1",
    entry_point="gym_minatar.freeway:Freeway",
    max_episode_steps=10_000,
)

register(
    id="Gym-MinAtar/Asterix-v1",
    entry_point="gym_minatar.asterix:Asterix",
    max_episode_steps=10_000,
)

register(
    id="Gym-MinAtar/Seaquest-v1",
    entry_point="gym_minatar.seaquest:Seaquest",
    max_episode_steps=10_000,
)

register(
    id="Gym-MinAtar/SpaceInvaders-v1",
    entry_point="gym_minatar.space_invaders:SpaceInvaders",
    max_episode_steps=10_000,
)
