import os
from glob import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


DATA_DIR = "ball_on_plate/data/raw_data"

SAVE_DIR = "ball_on_plate/data"


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


ACTION_COLS = ["roll_torque","pitch_torque",]

MIN_EPISODE_LENGTH = 50
TEST_SIZE = 0.15
RANDOM_SEED = 42

def load_csvs():

    files = sorted(glob(os.path.join(DATA_DIR,"*.csv")))

    print(f"Found {len(files)} CSV files")

    dfs=[]

    for f in files:

        df=pd.read_csv(f)

        dfs.append(df)

    data=pd.concat(dfs,ignore_index=True)

    print(f"Loaded {len(data):,} transitions")

    return data

def remove_short_episodes(df):

    lengths = (df.groupby("episode").size())

    valid_eps = lengths[lengths >= MIN_EPISODE_LENGTH].index

    filtered = df[df["episode"].isin(valid_eps)].copy()

    removed = (df["episode"].nunique()-filtered["episode"].nunique())

    print(f"Removed {removed} short episodes")

    return filtered

def create_transitions(df):

    states=[]
    actions=[]
    next_states=[]

    # Keep trajectories intact

    for ep, episode_df in df.groupby("episode"):

        episode_df = (episode_df.sort_values("step"))

        state = episode_df[STATE_COLS].values

        action = episode_df[ACTION_COLS].values

        # remove last timestep
        states.append(state[:-1])

        actions.append(action[:-1])

        next_states.append(state[1:])


    states=np.concatenate(states)
    actions=np.concatenate(actions)
    next_states=np.concatenate(next_states)


    print("\nTransitions created")
    print("States:",states.shape)
    print("Actions:",actions.shape)
    print("Next states:",next_states.shape)


    return states, actions, next_states


def split_by_episode(df):

    episodes = (df["episode"].unique())

    train_eps, test_eps = train_test_split(
        episodes,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
        )


    train_df=df[
        df.episode.isin(train_eps)
        ]

    test_df=df[
        df.episode.isin(test_eps)
        ]


    print("\nEpisode split")
    print("Train episodes:",len(train_eps))

    print("Test episodes:",len(test_eps))

    return train_df,test_df



def compute_statistics(states,actions):

    stats={"state_mean":states.mean(axis=0),"state_std":states.std(axis=0),"action_mean":actions.mean(axis=0),"action_std":actions.std(axis=0)}
    return stats



def main():

    os.makedirs(SAVE_DIR,exist_ok=True)

    df=load_csvs()

    df=remove_short_episodes(df)

    train_df,test_df = split_by_episode(df)

    train_data=create_transitions(train_df)

    test_data=create_transitions(test_df)

    train_states,train_actions,_ = train_data

    stats=compute_statistics(train_states,train_actions)
    
    np.savez(os.path.join(SAVE_DIR,"normalization.npz"),**stats)


    np.savez(os.path.join(SAVE_DIR,"train.npz"),state=train_data[0],action=train_data[1],next_state=train_data[2])


    np.savez(os.path.join(SAVE_DIR,"test.npz"),state=test_data[0],action=test_data[1],next_state=test_data[2])


    print("\nSaved datasets")

    print("Train transitions:",len(train_data[0]))

    print("Test transitions:",len(test_data[0]))


    print("\nState std:")
    print(stats["state_std"])

    print("\nAction std:")
    print(stats["action_std"])



if __name__=="__main__":
    main()