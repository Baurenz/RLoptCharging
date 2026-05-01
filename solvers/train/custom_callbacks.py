from stable_baselines3.common.callbacks import BaseCallback


class CustomMetricsCallback(BaseCallback):
    """
    This callback is used to log custom metrics for each episode during training.
    It tracks the sum of several cost components as well as the total reward.

    The callback logs the following metrics to TensorBoard:
    - The mean of each individual cost component per episode (`custom/{cost_component}_mean`).
    - The total reward per episode (`custom/total_reward_mean`), which is the sum of the individual cost components.
    - The mean of the total reward across all episodes so far (`custom/mean_reward_all_episodes`).

    The `custom/total_reward_mean` represents the sum of all cost components for a given episode.
    Since the individual costs are penalties and are negated (the more negative, the worse),
    the `total_reward_mean` is expected to be negative as well. If the training is improving,
    the magnitude of `custom/total_reward_mean` should increase (become less negative) over time.

    The individual costs are averaged per episode to give a sense of how each penalty component contributes
    to the total reward on average. These are visualized as separate plots for each penalty component.

    `custom/mean_reward_all_episodes` is the running average of the `total_reward_mean` across all episodes.
    This gives an overall trend of whether the agent's performance is improving over time.

    Attributes:
        verbose (int): The verbosity level: 0 for silent, 1 for interactive.
        total_penalties (dict): A dictionary that tracks the cumulative penalties for the current episode.
        episode_rewards (list): A list that stores the total reward for each episode.

    Methods:
        _on_step(): Called by Stable Baselines3 when a step is taken in the environment.
    """

    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.total_penalties = {
            'reward': 0,
            'penalty_lines_total': 0,
            'penalty_ev': 0,
            'cost_sum_bus_energy': 0,
            'wn_buying_ext_grid': 0,
            'wn_selling_ext_grid': 0,
            'penalty_sum_remaining_pv': 0,
            'sum_penalties_os_p_cs': 0,
            'penalty_sum_action_empty_cs': 0
        }
        self.episode_rewards = []
        self.aggregated_rewards = 0  # Total rewards accumulated over multiple episodes
        self.aggregated_lengths = 0  # Total length (number of timesteps) accumulated over multiple episodes
        self.aggregated_penalties = {key: 0 for key in self.total_penalties}  # Aggregated penalties for the window
        self.num_episodes = 0  # Number of episodes accumulated

    def _on_step(self) -> bool:
        # Accumulate penalties from the environment
        penalty_components = self.model.env.envs[0].return_partial_costs()
        for key in self.total_penalties.keys():
            if key in penalty_components:
                self.total_penalties[key] += penalty_components[key]

        if True in self.locals['dones']:
            episode_length = self.locals['infos'][0].get('episode_length', 1)
            total_reward_for_episode = sum(self.total_penalties.values())
            self.episode_rewards.append(total_reward_for_episode)

            # Aggregate rewards, lengths, and penalties
            self.aggregated_rewards += total_reward_for_episode
            self.aggregated_lengths += episode_length
            for key in self.total_penalties:
                self.aggregated_penalties[key] += self.total_penalties[key]
            self.num_episodes += 1

            # Check if aggregated lengths reach or exceed a certain threshold
            if self.aggregated_lengths >= 4032:
                mean_reward = self.aggregated_rewards / self.num_episodes
                mean_reward_all_episodes = sum(self.episode_rewards) / len(self.episode_rewards)

                self.logger.record('custom/01_total_reward_mean_4032', mean_reward)
                self.logger.record('custom/02_mean_reward_all_episodes', mean_reward_all_episodes)

                # Log the mean of each individual penalty cost for the window
                for penalty_name, penalty_total in self.aggregated_penalties.items():
                    if penalty_name in penalty_components:
                        if penalty_name == 'reward':
                            self.logger.record(f'custom/03_{penalty_name}_mean_4032', penalty_total / self.aggregated_lengths)
                        else:
                            self.logger.record(f'custom/{penalty_name}_mean_4032', penalty_total / self.aggregated_lengths)

                self.aggregated_rewards = 0
                self.aggregated_lengths = 0
                self.aggregated_penalties = {key: 0 for key in self.total_penalties}
                self.num_episodes = 0

            # Reset the total penalties for the next episode
            self.total_penalties = {key: 0 for key in self.total_penalties}

        return True

