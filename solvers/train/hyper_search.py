import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    # Other hyperparameters can be added here

    env = make_vec_env('YourEnvName', n_envs=4)
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, verbose=0)

    # Train the model
    model.learn(total_timesteps=100000)

    # Evaluate the model and return the metric of interest, e.g., cumulative reward
    eval_reward = evaluate_model(model)
    return eval_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Specify the number of trials

print("Best hyperparameters: ", study.best_params)
