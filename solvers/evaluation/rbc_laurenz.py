import argparse
import gymnasium as gym
import os
import time
from pathlib import Path
from RBC import RBC

import evaluate_soc_leave


from stable_baselines3.common.env_checker import check_env

import rl_OptV2GEnv.scenario.create_episode_profiles as create_profiles

debug_flag = False

##############
# SIMULATION CONFIG
##############
simnet = False

# Set this flag to the session number you want to continue training, or None to start a new session
session_prefix = "rbc_test_"
session_id = 1

# Scenario and network configuration paths
network_name = 'simple_CS_noLoad_4bus'

pp_network_path = f'../../data/scenarios/{network_name}_no_data.json'
network_config_path = f'../../create_data/profile_json/{network_name}.json'

parser = argparse.ArgumentParser(description="Setup for CsEnv")
parser.add_argument("--env", default="CsEnv-v0")
parser.add_argument('--price', type=int, default=0, help='Initial price setting')
parser.add_argument('--simnet', type=str, default=simnet, help='Simulation network identifier')
parser.add_argument('--solar', type=int, default=1, help='Solar setting')
parser.add_argument('--pp_network_path', type=str, default=pp_network_path, help='Path to the power network file')
parser.add_argument('--network_config_path', type=str, default=network_config_path, help='Path to the network configuration file')
parser.add_argument('--debug_flag', type=int, default=0, help='Debug mode flag')
parser.add_argument('--session_prefix', type=str, default=session_prefix, help='Prefix for session name')
parser.add_argument('--session_id', type=str, default=session_id, help='Session identifier')

args = parser.parse_args()

env = gym.make(args.env, price=args.price, simnet=args.simnet, solar=args.solar,
               pp_network_path=args.pp_network_path, network_config_path=args.network_config_path,
               debug_flag=args.debug_flag, session_name=f"{args.session_prefix}{args.session_id}",  eval_flag=True)

check_env(env)

rbc = RBC(env)

episodes = 5000
final_rewards = [0] * episodes
for ep in range(episodes):
    rewards_list = []
    done = False
    obs = env.reset(reset_flag=0)
    obs = obs[0]

    while not done:
        action = rbc.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)

        rewards_list.append(reward)
    final_rewards[ep] = sum(rewards_list)

env.close()


evaluate_soc_leave.evaluate_soc(session_prefix, session_id)


final_reward = sum(rewards_list)
print(final_reward)
