import os
import numpy as np
from oscillator.envs.harmonic_oscillator import HarmonicOscillatorEnv


EPISODES_PER_JOB = 1000
MAX_STEPS = 500

KP = 16
KD = 8

SAVE_DIR = "oscillator/data/raw"

os.makedirs(SAVE_DIR, exist_ok=True)

job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

print(f"Running job {job_id}")

start_episode = job_id * EPISODES_PER_JOB
end_episode = start_episode + EPISODES_PER_JOB

print(f"Episodes {start_episode} -> {end_episode-1}")




env = HarmonicOscillatorEnv(render_mode=None)

obs_list = []
actions_list = []
next_obs_list = []
dones_list = []


for ep in range(start_episode, end_episode):

    if ep % 50 == 0:
        print(f"Episode {ep}")

    obs, info = env.reset()
    target = info["target"]

    for step in range(MAX_STEPS):

        x, xdot, xddot = obs

        action = np.array(
            [KP * (target - x) - KD * xdot],
            dtype=np.float32,
        )

        next_obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append([x, xdot, xddot, target])
        actions_list.append(action[0])
        next_obs_list.append(
            [next_obs[0], next_obs[1], next_obs[2], target]
        )
        dones_list.append(terminated or truncated)

        obs = next_obs

        if terminated or truncated:
            break

env.close()

# =====================================================
# Save
# =====================================================

filename = os.path.join(
    SAVE_DIR,
    f"expert_data_{job_id:04d}.npz",
)

np.savez_compressed(
    filename,
    obs=np.asarray(obs_list, dtype=np.float32),
    actions=np.asarray(actions_list, dtype=np.float32),
    next_obs=np.asarray(next_obs_list, dtype=np.float32),
    dones=np.asarray(dones_list, dtype=bool),
)

print(f"Saved {filename}")
print(f"Transitions: {len(obs_list)}")