import argparse
import os, re
from stable_baselines3 import DDPG, PPO, A2C, TD3

from sbx import DDPG as DDPG_jax

from pathlib import Path


# Define your network name and other constants

def get_network_and_pp_path(network_name):
    # Configuration paths
    pp_network_path = f'data/pandappower_networks/{network_name}_pp.json'
    network_config_path = f'data/profile_json/{network_name}.json'

    return pp_network_path, network_config_path


# Argument parser setup
def get_parser(simnet, use_irradiance, use_real_load, objective_weights, pp_network_path, network_config_path, selected_model,
               session_prefix, session_id):
    parser = argparse.ArgumentParser(description="Setup for CsEnv")
    parser.add_argument("--env", default="OptV2GEnv-v0")
    parser.add_argument('--price', type=int, default=0, help='Initial price setting')
    parser.add_argument('--simnet', type=str, default=simnet, help='Simulation network identifier')
    parser.add_argument('--solar', type=int, default=1, help='Solar setting')
    parser.add_argument('--use_irradiance', type=bool, default=use_irradiance, help='using solar irradiance for pv profiles')
    parser.add_argument('--use_real_load', type=bool, default=use_real_load, help='using real load profiles')
    parser.add_argument('--objective_weights', type=dict, default=objective_weights, help='weights to control objective function')
    parser.add_argument('--pp_network_path', type=str, default=pp_network_path, help='Path to the power network file')
    parser.add_argument('--network_config_path', type=str, default=network_config_path, help='Path to the network configuration file')
    parser.add_argument('--debug_flag', type=int, default=0, help='Debug mode flag')
    parser.add_argument('--session_prefix', type=str, default=f'{selected_model}_{session_prefix}', help='Prefix for session name')
    parser.add_argument('--session_id', type=str, default=session_id, help='Session identifier')
    return parser


# Utility functions
def get_next_session_number(directory, session_prefix, session_id=None):
    """
    Gets the next session number or returns the provided session_id.

    Parameters:
    - directory (Path): The directory where session folders are stored.
    - session_prefix (str): The prefix used for naming session folders.
    - session_id (str/int, optional): The session identifier. If provided, this function returns it directly.

    Returns:
    - int: The next session number or the provided session_id.
    """
    if session_id is not None:
        return session_id  # Return the provided session_id directly

    # If session_id is not provided, calculate the next session number
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


def load_or_initialize_model(model_type, session_id, models_dir, env, logdir): #, action_noise, learning_rate=None, gamma=None):
    # Determine the model class based on the model_type string
    if model_type == "DDPG":
        ModelClass = DDPG
    elif model_type == "PPO":
        ModelClass = PPO
    elif model_type == "A2C":
        ModelClass = A2C
    elif model_type == "DDPG_jax":
        ModelClass = DDPG_jax
    elif model_type == "TD3":
        ModelClass = TD3
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Attempt to load the latest model, or initialize a new one
    if session_id is not None:
        latest_model_path = get_latest_model(models_dir)
        if latest_model_path:
            model = ModelClass.load(latest_model_path, env=env)
            print(f"Resuming training from: {latest_model_path}")
            match = re.search(r"(\d+)\.zip$", latest_model_path)
            if match:
                start_from = int(match.group(1))  # Access the first group and convert to int
        else:
            print("No saved model found in specified session. Starting new training in this session.")
            model = ModelClass("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))# action_noise=action_noise,
                              # learning_rate=learning_rate, gamma=gamma)
            start_from = 0
    else:
        model = ModelClass("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))#, action_noise=action_noise, learning_rate=learning_rate,
                           #gamma=gamma)
        start_from = 0

    return model, start_from
