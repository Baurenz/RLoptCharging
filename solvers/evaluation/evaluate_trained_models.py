import argparse
import json

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO, DDPG, A2C, TD3
from sbx import DDPG as DDPG_jax
from sb3_contrib import RecurrentPPO, ARS  # Import RecurrentPPO from sb3_contrib

## imports sys and sets root folder absolute. shoulf work on every system
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))
import rl_OptV2GEnv  # make sure this is imported
import evaluate_soc_leave
from solvers.train.config import get_network_and_pp_path, get_parser, get_next_session_number, get_latest_model, load_or_initialize_model
import check_results_evaluation_interactive

# Define which model to use: 'PPO', 'DDPG', or 'A2C'
model_type = 'DDPG'  # Change this to 'DDPG', 'RecurrentPPO' or 'A2C' as needed
# jax = False
# Define which model to load
session_prefix = 'scenariohigh'
model_episode = 580000
session_id = 1

simnet = False
use_irradiance = True
use_real_load = True

new_evaluation = True

base_models_dir = Path(f"solvers/models/{model_type}")
base_logdir = Path(f"solvers/results/logs/{model_type}")

model_path = Path(f"/home/laurenz/Documents/DAI/_Thesis/git/RLoptCharging/solvers/models/{model_type}/{session_prefix}{session_id}/{model_type}_{model_episode}")
config_path = model_path.parent / "config.json"  # Assuming the config file is in the same directory as the model

# Loading the configuration from the JSON file
with config_path.open() as json_file:
    config = json.load(json_file)

objective_weights = config['objective_weights']
network_name = config['network_name']

# Scenario and network configuration paths
pp_network_path, network_config_path = get_network_and_pp_path(network_name)

# TODO: maybe also store that in config. depending on how evaluation will look like maybe..
parser = get_parser(simnet, use_irradiance, use_real_load, objective_weights, pp_network_path, network_config_path, model_type,
                    session_prefix, session_id)
args = parser.parse_args()

env = gym.make(args.env, price=args.price, simnet=args.simnet, solar=args.solar,
               pp_network_path=args.pp_network_path, network_config_path=args.network_config_path, objective_weights=objective_weights,
               debug_flag=args.debug_flag, session_name=f"{args.session_prefix}{args.session_id}", eval_flag=True,
               use_irradiance=args.use_irradiance, use_real_load=args.use_real_load)

# Load the model based on the selected model type
if model_type == 'PPO':
    model = PPO.load(model_path, env=env)
elif model_type == 'DDPG':
    model = DDPG.load(model_path, env=env)
elif model_type == 'A2C':
    model = A2C.load(model_path, env=env)
elif model_type == 'RecurrentPPO':
    model = RecurrentPPO.load(model_path, env=env)
elif model_type == 'ARS':
    model = ARS.load(model_path, env=env)
elif model_type == 'DDPG_jax':
    model = DDPG_jax.load(model_path, env=env)
elif model_type == 'TD3':
    model = TD3.load(model_path, env=env)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# How many evaluations
if new_evaluation:
    episodes = 365
    final_rewards = [0] * episodes
    for ep in range(episodes):
        rewards_list = []

        obs, info = env.reset(reset_flag=2)     # reset_flag '0' evaluate random different days as in training
                                                # reset_flag '1' evaluate the same day
                                                # reset_flag '2' evaluates always the same pseudo random scenario, for same config, starting from 01.01.2019
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rewards_list.append(reward)

        final_rewards[ep] = sum(rewards_list)

    env.close()

# mean_reward = np.mean(final_rewards)
#
# plt.figure(figsize=(10, 6))  # Modify the values to suit your needs
# plt.rcParams.update({'font.size': 18})
# plt.plot(final_rewards)
# plt.xlabel('Evaluation episodes')
# plt.ylabel('Reward')
# plt.legend([selected_model])
#
# min_reward = min(final_rewards)
# worst_episode_index = final_rewards.index(min_reward)
#
# # Print the worst episode and its reward
# print(f"The worst episode is Episode {worst_episode_index} with a reward of {min_reward}")

# plt.show()


# evaluate_soc_leave.evaluate_soc(selected_model, session_prefix, session_id)

check_results_evaluation_interactive.run_interactive_analysis(model_type, session_prefix, session_id)
