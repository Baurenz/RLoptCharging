import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
import rl_charging_station
# from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines3 import DDPG, PPO
# from stable_baselines3.common.evaluation import evaluate_policy
import time

from sbx import TQC, DroQ, SAC, PPO, DQN, TD3, DDPG


# Record the start time
start_time = time.time()

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"results/logs/PPO-{int(time.time())}"

models_dir = Path(models_dir)
if not models_dir.exists():
    models_dir.mkdir(parents=True)

logdir = Path(logdir)
if not logdir.exists():
    logdir.mkdir(parents=True)

# pp_network_path = '../data/scenarios/4-Bus_4-PV_4-Load_4-V2G_externalgrid_bus1_no_data.json'


pp_network_path = '../../data/scenarios/Dynamic_Network_25_no_data.json'
network_config_path = '../../create_data/profile_json/dynamic_profile_25bus.json'

# pp_network_path = '../data/scenarios/easy_profile_5bus_no_data.json'
# network_config_path = '../create_data/profile_json/easy_profile_5bus.json'

# price is not a flag: 0 is price chart of real data, 1 is some random data

env = gym.make('CsEnv-v0', price=0, simnet=1, solar=1,
               pp_network_path=pp_network_path, network_config_path=network_config_path)
check_env(env)

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 20000

for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close()

# Record the end time
end_time = time.time()

# Calculate and print the time taken for training
time_taken = end_time - start_time
print(f"Total time taken for training: {time_taken} seconds")
