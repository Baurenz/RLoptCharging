from gymnasium.envs.registration import registry, register, make, spec

register(
     id='CsEnv-v0',
     entry_point='rl_charging_station.envs:CsEnv',
     max_episode_steps=200,
)
