import os
import glob
import pandas as pd


TRAJECTORY_DIR = "ball_on_plate/results/trajectories"
EPISODE_DIR = "ball_on_plate/results/episode_metrics"

OUTPUT_DIR = "ball_on_plate/results/merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def merge_files(pattern, output):

    files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} files")

    dfs = []

    for f in files:
        print("Loading:", f)
        dfs.append(pd.read_csv(f))

    merged = pd.concat(
        dfs,
        ignore_index=True
    )

    merged.to_csv(output,index=False)

    print("Saved:", output)


if __name__ == "__main__":
    diffs= ["easy","medium","hard","very_hard"]

    for diff in diffs:

        merge_files(f"{TRAJECTORY_DIR}/Expert_MPC_{diff}_job*.csv",f"{OUTPUT_DIR}/Expert_MPC_{diff}_trajectories.csv")

        merge_files(f"{EPISODE_DIR}/Expert_MPC_{diff}_job*.csv",f"{OUTPUT_DIR}/Expert_MPC_{diff}_metrics.csv")

    