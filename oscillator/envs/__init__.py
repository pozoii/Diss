from gymnasium.envs.registration import register

register(
    id="oscillator_env/GridWorld-v0",
    entry_point="oscillator_env.envs:GridWorldEnv",
)
