import os
from glob import glob

import numpy as np
import pandas as pd

DATA_DIR = "ball_on_plate/data"

STATE_COLS = [
    "x",
    "y",
    "xdot",
    "ydot",
    "alpha",
    "beta",
    "alphadot",
    "betadot",
]

ACTION_COLS = [
    "roll_torque",
    "pitch_torque",
]

# ------------------------------------------------------------------
# CHANGE THIS TO MATCH YOUR ENVIRONMENT
# ------------------------------------------------------------------
BALL_LIMIT = 0.25


def analyse_group(group_df):

    print(f"Rows: {len(group_df):,}")
    print(f"Episodes: {group_df['episode'].nunique()}")

    episode_lengths = group_df.groupby("episode").size()

    print("\nEpisode lengths")
    print(episode_lengths.describe())

    print("\nVery short episodes")
    print(f"<= 5 steps : {(episode_lengths <= 5).sum()}")
    print(f"<=10 steps : {(episode_lengths <= 10).sum()}")
    print(f"<=20 steps : {(episode_lengths <= 20).sum()}")
    print(f"<=50 steps : {(episode_lengths <= 50).sum()}")

    print("\nMissing values")
    missing = group_df.isnull().sum()
    print(missing[missing > 0])

    print("\nDone distribution")
    print(group_df["done"].value_counts())

    print("\nState statistics")
    print(group_df[STATE_COLS].describe().T[["mean", "std", "min", "max"]])

    print("\nAction statistics")
    print(group_df[ACTION_COLS].describe().T[["mean", "std", "min", "max"]])

    print("\nDuplicate rows")
    print(group_df.duplicated().sum())

    # ---------------------------------------------------------
    # Success / failure analysis
    # ---------------------------------------------------------

    last_rows = (
        group_df
        .sort_values(["episode", "step"])
        .groupby("episode")
        .tail(1)
    )

    ball_lost = (
        (last_rows["x"].abs() >= BALL_LIMIT)
        |
        (last_rows["y"].abs() >= BALL_LIMIT)
    )

    successes = (~ball_lost).sum()
    failures = ball_lost.sum()

    print("\nTermination")
    print(f"Successful stabilisations : {successes}")
    print(f"Ball lost                : {failures}")
    print(f"Success rate             : {100*successes/len(last_rows):.2f}%")
    print(f"Failure rate             : {100*failures/len(last_rows):.2f}%")

    return episode_lengths


def inspect_dataset(data_dir):

    files = sorted(glob(os.path.join(data_dir, "*.csv")))

    print(f"Found {len(files)} CSV files")

    groups = {}

    for file in files:

        df = pd.read_csv(file)

        key = (
            df["max_pos_reset"].iloc[0],
            df["max_vel_reset"].iloc[0],
        )

        groups.setdefault(key, []).append(df)

    print(f"\nFound {len(groups)} difficulty groups.\n")

    all_dfs = []

    for (max_pos, max_vel), dfs in sorted(groups.items()):

        group_df = pd.concat(dfs, ignore_index=True)

        all_dfs.append(group_df)

        print("=" * 70)
        print(f"max_pos_reset = {max_pos}")
        print(f"max_vel_reset = {max_vel}")
        print("=" * 70)

        analyse_group(group_df)

        print("\n")

    # ---------------------------------------------------------

    full_df = pd.concat(all_dfs, ignore_index=True)

    print("=" * 70)
    print("GLOBAL DATASET")
    print("=" * 70)

    analyse_group(full_df)


if __name__ == "__main__":
    inspect_dataset(DATA_DIR)