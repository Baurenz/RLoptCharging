# ddpg_train_continue.py
import argparse
import json
import gymnasium as gym
import time
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
# for the following to work from project root, project_root will be appended to sys.path
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
import sys

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))
from network_env.create_pandapower_network_dynamic import create_pp_network
import rl_OptV2GEnv
from custom_callbacks import CustomMetricsCallback
from config import get_network_and_pp_path, get_parser, get_next_session_number, get_latest_model, load_or_initialize_model

# Use the custom callback

##################
# TRAINING CONFIG
################
# Set this flag to the session number you want to continue training, or None to start a new session
model_type = 'A2C'
session_prefix = "PRES"
# Scenario and network configuration paths
network_name = 'simple_CS_Load_v2g_ESS_4bus'
session_id = '55'
debug_flag = False

objective_weights = {'wm_cs': 40,
                     'wm_ep_b': 1,  # weight for energy price buying
                     'wm_ep_s': 1,
                     'wm_rp': 0,  # weight for remaining_pv
                     'wm_os_c': 0,
                     'wm_c_sum_be': 0,  # cost sum bus energy, prob not needed
                     'wm_empty_station': 0,
                     'wm_l': 0}

# runs a pandapower simulation in every timestep, influences reward function and shifts value source of energy calculations to pp results
simnet = False
use_irradiance = True
use_real_load = True
###########################
##########################
#########################

create_pp_network(network_name)
pp_network_path, network_config_path = get_network_and_pp_path(network_name)

parser = get_parser(simnet, use_irradiance, use_real_load, objective_weights, pp_network_path, network_config_path, model_type,
                    session_prefix, session_id)
args = parser.parse_args()

start_time = time.time()

base_models_dir = Path(f"solvers/models/{model_type}")
base_logdir = Path(f"solvers/results/logs/{model_type}")

# Determine the session number
session_number = get_next_session_number(base_models_dir, session_prefix, session_id)

session_name = f"{session_prefix}{session_number}"
models_dir = base_models_dir / session_name
logdir = base_logdir / session_name

# Create directories if they do not exist
models_dir.mkdir(parents=True, exist_ok=True)
logdir.mkdir(parents=True, exist_ok=True)


config = {'network_name': network_name,
          'objective_weights': objective_weights}

json_path = models_dir / "config.json"
with json_path.open("w") as json_file:
    json.dump(config, json_file, indent=4)


# Environment setup
env = gym.make(args.env, price=args.price, simnet=args.simnet, solar=args.solar,
               pp_network_path=args.pp_network_path, network_config_path=args.network_config_path, objective_weights=objective_weights,
               debug_flag=args.debug_flag, session_name=f"{args.session_prefix}{args.session_id}",
               use_irradiance=args.use_irradiance, use_real_load=args.use_real_load)

check_env(env)

if model_type == 'PPO':
    n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    action_noise = None
    learning_rate = 0.001
    gamma = 0.99
else:
    action_noise = None

# Load the latest model if continuing training
model, start_from = load_or_initialize_model(model_type, args.session_id, models_dir, env, logdir)#, action_noise, learning_rate, gamma)
callback = CustomMetricsCallback()
# Training loop
TIMESTEPS = 20000
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG", callback=callback)
    model.save(f"{models_dir}/{model_type}_{start_from + TIMESTEPS * i}")
    # create_profiles.create_new_profiles(network_config_path, use_irradiance)

env.close()

# Record the end time and print the time taken for training
end_time = time.time()
time_taken = end_time - start_time
print(f"Total time taken for training: {time_taken} seconds")
