from gymnasium.envs.registration import register

register(
    id='discrete_env/GridWorld-v0',
    entry_point='discrete_env.envs:GridWorldEnv',
)
