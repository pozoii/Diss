import os
import numpy as np
import pandas as pd
from joblib import load

from ball_on_plate.envs.ball_on_plate import BallOnPlateEnv
from ball_on_plate.expert.expert_MPC import ExpertMPC


EPISODES_PER_JOB = 100

SAVE_EVERY = 5

HORIZON = 50

MAX_STEPS = 5000

SETTLING_WINDOW = 50

SAVE_DIR = "ball_on_plate/data"

os.makedirs(SAVE_DIR, exist_ok=True)

job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

print(f"Running job {job_id}")

difficulty_level = job_id % 4


difficulty = {
    0: {"max_pos_reset": 0.15, "max_vel_reset": 0.50,},
    1: {"max_pos_reset": 0.20,"max_vel_reset": 0.50,},
    2: {"max_pos_reset": 0.20,"max_vel_reset": 0.80, },
    3: {"max_pos_reset": 0.24,"max_vel_reset": 1.0,},
    }[difficulty_level]


print("Difficulty:")
print(difficulty)


start_episode = job_id * EPISODES_PER_JOB

end_episode = start_episode + EPISODES_PER_JOB

print(f"Episodes {start_episode} -> {end_episode-1}")

# =====================================================
# Environment + Expert
# =====================================================

env = BallOnPlateEnv(render_mode=None,max_pos_reset=difficulty["max_pos_reset"],max_vel_reset=difficulty["max_vel_reset"], settling_window=SETTLING_WINDOW)

plate_models = load("ball_on_plate/dynamics/plate_models.joblib")

plate_models = (plate_models["roll_model"],plate_models["pitch_model"],)

expert = ExpertMPC(env=env,plate_models=plate_models,H=HORIZON,)

filename = os.path.join(SAVE_DIR,f"expert_data_{job_id:04d}.csv")

buffer = []

for ep in range(start_episode, end_episode):

    print(f"Job {job_id} | Episode {ep}")

    obs, info = env.reset()

    expert.reset()

    init_x = obs[0]
    init_y = obs[1]
    init_xdot = obs[2]
    init_ydot = obs[3]

    for step in range(MAX_STEPS):

        action = expert.control(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)


        done = terminated or truncated


        record = {
            # identification
            "job_id": job_id,
            "episode": ep,
            "step": step,

            # initial condition
            "init_x": init_x,
            "init_y": init_y,
            "init_xdot": init_xdot,
            "init_ydot": init_ydot,

            # current state
            "x": obs[0],
            "y": obs[1],
            "xdot": obs[2],
            "ydot": obs[3],
            "alpha": obs[4],
            "beta": obs[5],
            "alphadot": obs[6],
            "betadot": obs[7],

            # action
            "roll_torque": action[0],
            "pitch_torque": action[1],

            # next state
            "x_next": next_obs[0],
            "y_next": next_obs[1],
            "xdot_next": next_obs[2],
            "ydot_next": next_obs[3],
            "alpha_next": next_obs[4],
            "beta_next": next_obs[5],
            "alphadot_next": next_obs[6],
            "betadot_next": next_obs[7],

            # termination
            "done": done,

            # generation metadata
            "H": HORIZON,
            "max_pos_reset": difficulty["max_pos_reset"],
            "max_vel_reset": difficulty["max_vel_reset"],
        }

        buffer.append(record)


        obs = next_obs


        if done:
            break

    # ---------------------------------------------
    # Periodic checkpoint
    # ---------------------------------------------

    if (ep - start_episode + 1) % SAVE_EVERY == 0:

        print(f"Saving checkpoint ({len(buffer)} rows)")

        df = pd.DataFrame(buffer)

        df.to_csv(filename,mode="a",header=not os.path.exists(filename),index=False,)

        buffer = []


# =====================================================
# Save remaining data
# =====================================================

if len(buffer) > 0:

    print(f"Saving final checkpoint ({len(buffer)} rows)")

    df = pd.DataFrame(buffer)

    df.to_csv(filename,mode="a",header=not os.path.exists(filename),index=False,)

env.close()

print("Finished.")º