import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3.common.env_checker import check_env

import rl_charging_station
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import time

models_dir = f"models/DDPG-{int(time.time())}"
logdir = f"results/logs/DDPG-{int(time.time())}"

models_dir = Path(models_dir)
if not models_dir.exists():
    models_dir.mkdir(parents=True)

logdir = Path(logdir)
if not logdir.exists():
    logdir.mkdir(parents=True)

env = gym.make('CsEnv-v0', price=1, solar=1)  # price=0 takes real pricing from ./data/market
check_env(env)

# observation, info = env.reset(seed=42)

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# DDPG model
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=logdir, action_noise=action_noise)

TIMESTEPS = 20000

for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close()
