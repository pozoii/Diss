import os
import pandas as pd
import numpy as np
from glob import glob


DATA_DIR = "ball_on_plate/expert"

STATE_COLS = [
    "x",
    "y",
    "xdot",
    "ydot",
    "alpha",
    "beta",
    "alphad",
    "betad",
]

ACTION_COLS = [
    "roll_torque",
    "pitch_torque",
]


def inspect_dataset(data_dir):

    files = sorted(glob(os.path.join(data_dir, "*.csv")))

    print(f"Found {len(files)} CSV files")

    all_dfs = []

    for i, file in enumerate(files):

        print("\n" + "="*60)
        print(f"File {i+1}/{len(files)}")
        print(file)

        df = pd.read_csv(file)

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        print(list(df.columns))

        print("\nMissing values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])

        print("\nEpisodes:")
        print(df["episode"].nunique())

        print("\nSteps per episode:")
        print(
            df.groupby("episode")
            .size()
            .describe()
        )

        print("\nDifficulty:")
        if "difficulty" in df.columns:
            print(df["difficulty"].value_counts())
        else:
            print("No difficulty column")

        print("\nDone distribution:")
        if "done" in df.columns:
            print(df["done"].value_counts())
        else:
            print("No done column")


        all_dfs.append(df)


    full_df = pd.concat(all_dfs, ignore_index=True)

    print("\n\n")
    print("="*60)
    print("GLOBAL DATASET SUMMARY")
    print("="*60)


    print("\nShape:")
    print(full_df.shape)


    print("\nColumns missing:")
    
    expected = STATE_COLS + ACTION_COLS + ["episode","step"]

    for c in expected:
        if c not in full_df.columns:
            print("MISSING:",c)


    print("\nState statistics:")
    print(
        full_df[STATE_COLS]
        .describe()
        .T
    )


    print("\nAction statistics:")
    print(
        full_df[ACTION_COLS]
        .describe()
        .T
    )


    print("\nDuplicate rows:")
    print(full_df.duplicated().sum())


    return full_df



if __name__ == "__main__":

    df = inspect_dataset(DATA_DIR)