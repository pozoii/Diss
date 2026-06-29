import numpy as np
import os

# -----------------------------
# Load full dataset
# -----------------------------
data = np.load("oscillator/data/raw_expert_pd_kp32_kd2.npz")

obs = data["obs"]
actions = data["actions"]
next_obs = data["next_obs"]
dones = data["dones"].astype(bool)

# -----------------------------
# Build episode IDs
# -----------------------------
episode_ids = np.cumsum(dones.astype(int))
num_episodes = episode_ids.max() + 1

print(f"Total transitions: {len(obs)}")
print(f"Total episodes: {num_episodes}")

# -----------------------------
# Shuffle episodes (IMPORTANT: not transitions)
# -----------------------------
rng = np.random.default_rng(42)
episode_order = rng.permutation(num_episodes)

# -----------------------------
# Split episodes
# -----------------------------
train_ratio = 0.8


train_end = int(train_ratio * num_episodes)

train_eps = set(episode_order[:train_end])
val_eps   = set(episode_order[train_end:])

# -----------------------------
# Mask transitions by episode
# -----------------------------
train_mask = np.isin(episode_ids, list(train_eps))
val_mask   = np.isin(episode_ids, list(val_eps))

# -----------------------------
# Helper to save split
# -----------------------------
def save_split(name, mask):
    np.savez(
        f"oscillator/data/{name}.npz",
        obs=obs[mask],
        actions=actions[mask],
        next_obs=next_obs[mask],
        dones=dones[mask],
    )
    print(f"{name}: {mask.sum()} transitions")

# -----------------------------
# Save files
# -----------------------------
save_split("train", train_mask)
save_split("val", val_mask)

print("\nDone. Splits saved in data folder.")