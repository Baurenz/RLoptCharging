import os
import gymnasium as gym
import numpy as np
from pathlib import Path
import rl_OptV2GEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
import rl_OptV2GEnv.scenario.create_episode_profiles as create_profiles
import time


def make_env(pp_network_path, network_config_path, rank, seed=0, log_dir=None):
    """
    Utility function for multiprocessed env.

    :param pp_network_path: The path to the power network scenario file
    :param network_config_path: The path to the network configuration file
    :param rank: the index of the subprocess env
    :param seed: the initial seed for RNG
    :param log_dir: the log directory for the Monitor
    :return: the environment
    """

    def _init():
        env = gym.make('CsEnv-v0', price=0, simnet=1, solar=1,
                       pp_network_path=pp_network_path,
                       network_config_path=network_config_path)
        env.seed(seed + rank)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env

    return _init


def main():
    start_time = time.time()

    models_dir = Path(f"models/PPO-{int(time.time())}")
    logdir = Path(f"results/logs/PPO-{int(time.time())}")
    models_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    pp_network_path = '../../data/scenarios/easy_profile_2bus_no_data.json'
    network_config_path = '../../create_data/profile_json/easy_profile_2bus.json'

    create_profiles.create_new_profiles(network_config_path)

    num_cpu = 2  # Number of parallel environments
    envs = [make_env(pp_network_path, network_config_path, i, log_dir=logdir) for i in range(num_cpu)]
    env = SubprocVecEnv(envs)

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))

    TIMESTEPS = 20000
    for i in range(1, 50):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS * i}")
        create_profiles.create_new_profiles(network_config_path)

    env.close()

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Total time taken for training: {time_taken} seconds")


if __name__ == '__main__':
    main()
