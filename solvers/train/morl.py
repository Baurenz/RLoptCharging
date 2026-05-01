import argparse
import gymnasium as gym
import os
import time
from pathlib import Path
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
import rl_OptV2GEnv
import rl_OptV2GEnv.scenario.create_episode_profiles as create_profiles

from custom_callbacks import CustomMetricsCallback

# from sbx import TQC, DroQ, SAC, PPO, DQN, TD3, DDPG

# Use the custom callback
callback = CustomMetricsCallback()
debug_flag = False

# Set this flag to the session number you want to continue training, or None to start a new session
session_prefix = "check_res_pv_penaltyos"
session_id = 1

##############
# SIMULATION CONFIG
##############
simnet = False
use_irradiance = True
use_real_load = True

# Scenario and network configuration paths
network_name = 'PV_V2G_Load_4bus_simple'

pp_network_path = f'../../data/scenarios/{network_name}_no_data.json'
network_config_path = f'../../create_data/profile_json/{network_name}.json'

parser = argparse.ArgumentParser(description="Setup for CsEnv")
parser.add_argument("--env", default="CsEnv-v0")
parser.add_argument('--price', type=int, default=0, help='Initial price setting')
parser.add_argument('--simnet', type=str, default=simnet, help='Simulation network identifier')
parser.add_argument('--solar', type=int, default=1, help='Solar setting')
parser.add_argument('--use_irradiance', type=bool, default=use_irradiance, help='using solar irradiance for pv profiles')
parser.add_argument('--use_real_load', type=bool, default=use_real_load, help='using real load profiles')
parser.add_argument('--pp_network_path', type=str, default=pp_network_path, help='Path to the power network file')
parser.add_argument('--network_config_path', type=str, default=network_config_path, help='Path to the network configuration file')
parser.add_argument('--debug_flag', type=int, default=0, help='Debug mode flag')
parser.add_argument('--session_prefix', type=str, default=session_prefix, help='Prefix for session name')
parser.add_argument('--session_id', type=str, default=session_id, help='Session identifier')
# parser.add_argument('--morl', type= bolean)

args = parser.parse_args()

start_time = time.time()

base_models_dir = Path("../models/DDPG")
base_logdir = Path("../results/logs/DDPG")


def get_next_session_number(directory, session_prefix):
    existing_sessions = [d.name for d in directory.iterdir() if d.is_dir() and d.name.startswith(session_prefix)]
    if not existing_sessions:
        return 1
    else:
        return max([int(s.split('-')[-1]) for s in existing_sessions]) + 1


def get_latest_model(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        return None
    latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    return os.path.join(models_dir, latest_model_file)


# Determine the session number
if session_id is None:
    session_number = get_next_session_number(base_models_dir, session_prefix)
else:
    session_number = session_id

session_name = f"{session_prefix}{session_number}"
models_dir = base_models_dir / session_name
logdir = base_logdir / session_name

# Create directories if they do not exist
models_dir.mkdir(parents=True, exist_ok=True)
logdir.mkdir(parents=True, exist_ok=True)

if not debug_flag:
    create_profiles.create_new_profiles(network_config_path, use_irradiance)

# Environment setup
env = gym.make(args.env, price=args.price, simnet=args.simnet, solar=args.solar,
               pp_network_path=args.pp_network_path, network_config_path=args.network_config_path,
               debug_flag=args.debug_flag, session_name=f"{args.session_prefix}{args.session_id}",
               use_irradiance=args.use_irradiance, use_real_load=args.use_real_load)

check_env(env)

# Load the latest model if continuing training
if session_id is not None:
    latest_model_path = get_latest_model(models_dir)
    if latest_model_path:
        model = DDPG.load(latest_model_path, env=env)
        print(f"Resuming training from: {latest_model_path}")
    else:
        print("No saved model found in specified session. Starting new training in this session.")
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
else:
    model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))

# Training loop
TIMESTEPS = 20000
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG", callback=callback)
    model.save(f"{models_dir}/DDPG_{TIMESTEPS * i}")
    create_profiles.create_new_profiles(network_config_path, use_irradiance)

env.close()

# Record the end time and print the time taken for training
end_time = time.time()
time_taken = end_time - start_time
print(f"Total time taken for training: {time_taken} seconds")
