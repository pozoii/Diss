from gymnasium.envs.registration import register

register(
    id="ball_on_plate_env/GridWorld-v0",
    entry_point="ball_on_plate_env.envs:GridWorldEnv",
)