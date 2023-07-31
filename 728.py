import stable_baselines3 as sb3
import numpy as np
import mo_gymnasium as mo_gym


env = mo_gym.make('LunarLander-v2')

# Then, call the learn method with appropriate parameters
# For example, to train the agent for 10,000 timesteps:
agent = sb3.DQN(policy="MlpPolicy",
                env=env,
                verbose=1,
                tensorboard_log="./tensorboard/LunarLander-v2/",
                learning_rate=5e-4,
                weight=np.array([0.7, 0.1, 0.1, 0.1]))
agent.learn(total_timesteps=10000)

