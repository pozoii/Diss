"""

import pandas as pd
import re
from pathlib import Path

results_dir = Path("ball_on_plate/expert/mpc_horizons")  

for raw_file in results_dir.glob("H_*.csv"):

    if raw_file.stem.startswith("metrics"):
        continue

    H = re.search(r"H_(\d+)", raw_file.stem).group(1)
    
    metrics_file = results_dir / f"metrics_H_{H}.csv"

    if not metrics_file.exists():
        print(f"Missing: {metrics_file}")
        continue

    df = pd.read_csv(raw_file)

    episode_max_steps = df.groupby("episode")["step"].max()

    failure_rate = (episode_max_steps < 4999).mean()

    metrics_df = pd.read_csv(metrics_file)
    metrics_df["failure_rate"] = failure_rate

    metrics_df.to_csv(metrics_file, index=False)

    print(f"H={H}: failure_rate={failure_rate:.3f}")

    

print("Done")
"""


import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(df, target=np.array([0.0, 0.0]),settling_threshold=0.02,dt=0.002):

    metrics = {}

    pos_error = np.sqrt((df["x"] - target[0])**2 +(df["y"] - target[1])**2)

    metrics["mean_position_error"] = pos_error.mean()
    metrics["max_position_error"] = pos_error.max()

    torque = np.sqrt(df["roll_torque"]**2 +df["pitch_torque"]**2)

    metrics["mean_torque"] = torque.mean()
    metrics["control_cost"] = np.sum(torque**2)

    episode_lengths = []
    failures = []
    settling_times = []

    for _, ep_df in df.groupby("episode"):

        episode_lengths.append(len(ep_df))

        failures.append(len(ep_df) != 5000)

        error = np.sqrt(ep_df["x"]**2 + ep_df["y"]**2).values

        time = np.nan
        window = 100

        for i in range(len(error) - window):
            if np.all(error[i:i+window] < settling_threshold):
                time = i * dt
                break

        settling_times.append(time)

    metrics["mean_episode_length"] = np.mean(episode_lengths)
    metrics["failure_rate"] = np.mean(failures)
    metrics["mean_settling_time"] = np.nanmean(settling_times)
    metrics["computation_time_per1000"] = (df["computation_time"].iloc[0] / len(df) * 1000)
    metrics["H"]= df["H"].iloc[0]

    return metrics

def plot_metric(metric, ylabel, filename):

    plt.figure(figsize=(6,4))

    plt.plot(metrics["H"], metrics[metric], "o-", linewidth=2)

    plt.xscale("log")

    plt.xlabel("Prediction Horizon H")
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()



DIR = "ball_on_plate/expert/mpc_horizons"
METRICS_DIR = os.path.join(DIR, "metrics")
FIG_DIR = os.path.join(DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

trajectory_files = sorted(glob.glob(os.path.join(DIR, "H_*.csv")))

metrics = []

for f in trajectory_files:

    df = pd.read_csv(f)

    H = int(df["H"].iloc[0])

    m = compute_metrics(df)
    m["H"] = H


    pd.DataFrame([m]).to_csv(os.path.join(METRICS_DIR, f"metrics_H_{H}.csv"),index=False)

    metrics.append(m)

metrics = (pd.DataFrame(metrics).sort_values("H").reset_index(drop=True))
metrics.to_csv(os.path.join(METRICS_DIR, "summary.csv"),index=False,)

print("\n==============================")
print("MPC Horizon Summary")
print("==============================")
print(metrics)

plot_metric("mean_position_error","Mean position error (m)","mean_position_error.png",)

plot_metric("max_position_error","Max position error (m)","max_position_error.png")

plot_metric("mean_settling_time","Mean settling time (s)","settling_time.png")

plot_metric("control_cost","Control cost","control_cost.png",)

plot_metric("mean_torque","Mean torque","mean_torque.png",)

plot_metric("failure_rate","Failure rate","failure_rate.png",)

plot_metric("computation_time_per1000","Seconds per 1000 steps","computation_time.png",)

# ---------------------------------------------------
# Pareto plot
# ---------------------------------------------------

plt.figure(figsize=(6,5))

plt.scatter(metrics["computation_time_per1000"],metrics["mean_position_error"],s=80,)

for _, row in metrics.iterrows():

    plt.text(row["computation_time_per1000"],row["mean_position_error"],str(int(row["H"])),fontsize=9,)

plt.xlabel("Computation time /1000 steps (s)")
plt.ylabel("Mean position error (m)")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pareto.png"), dpi=300)
plt.close()


# ---------------------------------------------------
# Print best horizons
# ---------------------------------------------------

print("\n==============================")
print("Best horizons")
print("==============================")

print("\nLowest mean position error:")
print(metrics.loc[metrics["mean_position_error"].idxmin()])

print("\nLowest settling time:")
print(metrics.loc[metrics["mean_settling_time"].idxmin()])

print("\nLowest computation time:")
print(metrics.loc[metrics["computation_time_per1000"].idxmin()])

print("\nLowest control cost:")
print(metrics.loc[metrics["control_cost"].idxmin()])