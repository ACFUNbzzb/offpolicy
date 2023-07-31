import stable_baselines3 as sb3
import numpy as np
import mo_gymnasium as mo_gym

# Linear scalarizes the environment
env = mo_gym.LinearReward(mo_gym.make("mo-lunar-lander-v2"), weight=np.array([0.7, 0.1, 0.1, 0.1]))

# Run DQN agent!
agent = sb3.DQN(policy="MlpPolicy",
                env=env,
                verbose=1,
                tensorboard_log="./tensorboard/LunarLander-v2/",
                learning_rate=5e-4,
                learning_starts=0,
                )
agent.learn(10000)

