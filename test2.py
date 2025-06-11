import gym
import time

id = "MinAtar/Freeway-v1"

t1 = time.time()
env = gym.make(id, render_mode="human")
obs = env.reset()
env.render()

for i in range(100):
    obs, _, term, *_ = env.step(env.action_space.sample())
    if term: obs = env.reset()
    env.render()

t2 = time.time()
print(t2 - t1)

env.close()
