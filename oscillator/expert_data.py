
import numpy as np
from tqdm import tqdm  
from oscillator.envs.harmonic_oscillator import HarmonicOscillatorEnv

def collect_expert_data(Kp=8, Kd=2, episodes=200, max_steps=500):

    env = HarmonicOscillatorEnv(render_mode=None)

    all_obs = []
    all_actions = []
    all_next_obs = []
    all_dones = []

    for ep in tqdm(range(episodes), desc="Collecting expert episodes"):

        obs, info = env.reset()
        target = info["target"]

        for _ in range(max_steps):

            x, xdot, xdotdot = obs

            action = np.array([Kp * (target - x) - Kd * xdot], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)

            all_obs.append([x, xdot, xdotdot, target])
            all_actions.append(action[0])
            all_next_obs.append([next_obs[0], next_obs[1], next_obs[2], target])
            all_dones.append(terminated or truncated)

            obs = next_obs

            if terminated or truncated:
                break

    env.close()

    return {
        "obs": np.array(all_obs),
        "actions": np.array(all_actions),
        "next_obs": np.array(all_next_obs),
        "dones": np.array(all_dones),
    }

data = collect_expert_data(32,2,100000,500)

np.savez(
    "oscillator/data/raw_expert_pd_kp32_kd2.npz",
    obs=data["obs"],
    actions=data["actions"],
    next_obs=data["next_obs"],
    dones=data["dones"]
)