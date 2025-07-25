from gymnasium.envs.registration import register

register(
    id="embers_env/DiscreteWorld-v0",
    entry_point="embers_env.envs:DiscreteWorldEnv",
)

register(
    id="embers_env/ContinuousWorld-v0",
    entry_point="embers_env.envs:ContinuousWorldEnv",
)