import numpy as np
from joblib import load
import time
import sys
import os
import pandas as pd

from ball_on_plate.expert.expert_MPC import run_ExpertMPC


def main():

    Hs = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 250]

    # Number of jobs assigned to each horizon
    repeats = [1, 1, 1, 1, 3, 5, 9, 13, 17, 24, 25]

    if len(sys.argv) > 1:
        job_id = int(sys.argv[1])
    else:
        job_id = 0

    # Map job_id -> horizon and repetition index
    cumulative = 0

    for h_idx, n_jobs in enumerate(repeats):
        if job_id < cumulative + n_jobs:
            repeat_idx = job_id - cumulative
            break
        cumulative += n_jobs

    H = Hs[h_idx]

    print(f"SLURM job {job_id}")
    print(f"Running MPC with horizon H={H}, repeat={repeat_idx}")


    start_time = time.perf_counter()

    df = run_ExpertMPC(
        H=H,
        episodes=25,
        max_steps=5000
    )

    total_time = time.perf_counter() - start_time

    df["computation_time"] = total_time
    df["H"] = H
    df["repeat"] = repeat_idx
    df["job_id"] = job_id


    save_dir = "ball_on_plate/expert/mpc_horizons"
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(
        save_dir,
        f"H_{H}_run_{repeat_idx}.csv"
    )

    df.to_csv(filename, index=False)

    print(f"Saved {filename}")
    print(f"Computation time: {total_time:.2f}s")


if __name__ == "__main__":
    main()