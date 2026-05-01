from gymnasium.envs.registration import registry, register, make, spec

register(
     id='OptV2GEnv-v0',
     entry_point='rl_OptV2GEnv.envs:OptV2GEnv',
     max_episode_steps=200,
)
