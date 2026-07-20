
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIR = "ball_on_plate/expert"
RESULTS_DIR = os.path.join(DIR, "mpc_horizons")
FIG_DIR = os.path.join(DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


metric_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_H_*.csv")))

metrics = []

for f in metric_files:
    metrics.append(pd.read_csv(f))

metrics = pd.concat(metrics, ignore_index=True)
metrics = metrics.sort_values("H").reset_index(drop=True)

print("\n==============================")
print("MPC Horizon Summary")
print("==============================")
print(metrics)

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


plot_metric(
    "mean_position_error",
    "Mean position error (m)",
    "mean_position_error.png",
)

plot_metric(
    "max_position_error",
    "Max position error (m)",
    "max_position_error.png",
)

plot_metric(
    "mean_settling_time",
    "Mean settling time (s)",
    "settling_time.png",
)

plot_metric(
    "control_cost",
    "Control cost",
    "control_cost.png",
)

plot_metric(
    "mean_torque",
    "Mean torque",
    "mean_torque.png",
)

plot_metric(
    "failure_rate",
    "Failure rate",
    "failure_rate.png",
)

plot_metric(
    "computation_time_per1000",
    "Seconds per 1000 steps",
    "computation_time.png",
)


# ---------------------------------------------------
# Pareto plot
# ---------------------------------------------------

plt.figure(figsize=(6,5))

plt.scatter(
    metrics["computation_time_per1000"],
    metrics["mean_position_error"],
    s=80,
)

for _, row in metrics.iterrows():
    plt.text(
        row["computation_time_per1000"],
        row["mean_position_error"],
        str(int(row["H"])),
        fontsize=9,
    )

plt.xlabel("Computation time /1000 steps (s)")
plt.ylabel("Mean position error (m)")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pareto.png"), dpi=300)
plt.close()


# ---------------------------------------------------
# Trajectory comparison
# ---------------------------------------------------

trajectory_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "H_*.csv")))

plt.figure(figsize=(8,5))

for f in trajectory_files:

    df = pd.read_csv(f)

    ep0 = df[df["episode"] == 0]

    error = np.sqrt(ep0["x"]**2 + ep0["y"]**2)

    plt.plot(error.values, label=f"H={int(ep0['H'].iloc[0])}")

plt.xlabel("Step")
plt.ylabel("Radial error (m)")
plt.legend(ncol=2)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "trajectory_error.png"), dpi=300)
plt.close()


# ---------------------------------------------------
# Control comparison
# ---------------------------------------------------

plt.figure(figsize=(8,5))

for f in trajectory_files:

    df = pd.read_csv(f)

    ep0 = df[df["episode"] == 0]

    torque = np.sqrt(
        ep0["roll_torque"]**2 +
        ep0["pitch_torque"]**2
    )

    plt.plot(torque.values, label=f"H={int(ep0['H'].iloc[0])}")

plt.xlabel("Step")
plt.ylabel("Torque magnitude")
plt.legend(ncol=2)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "torque.png"), dpi=300)
plt.close()


# ---------------------------------------------------
# Print best horizons
# ---------------------------------------------------

print("\n==============================")
print("Best horizons")
print("==============================")

print(
    "\nLowest mean position error:"
)
print(
    metrics.loc[
        metrics["mean_position_error"].idxmin()
    ]
)

print(
    "\nLowest settling time:"
)
print(
    metrics.loc[
        metrics["mean_settling_time"].idxmin()
    ]
)

print(
    "\nLowest computation time:"
)
print(
    metrics.loc[
        metrics["computation_time_per1000"].idxmin()
    ]
)

print(
    "\nLowest control cost:"
)
print(
    metrics.loc[
        metrics["control_cost"].idxmin()
    ]
)