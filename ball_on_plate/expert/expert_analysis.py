import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POSITION_THRESHOLD = 0.02      # metres
VELOCITY_THRESHOLD = 0.02      # m/s
SETTLING_WINDOW = 50          # consecutive timesteps
MAX_EPISODE_LENGTH = 5000


H= 50
DF_DIR = f"ball_on_plate/expert/mpc_horizons/H_{H}.csv"
df = pd.read_csv(DF_DIR)

total_timesteps = len(df)

episode_lengths = []

df["concat_settle"] = 0
df["T_settled"] = 0

false_settles=0
settles=0
settling_step=[]

for _, ep_df in df.groupby("episode"):
    
    episode_lengths.append(len(ep_df))

    ep_df['position_error'] = np.sqrt(ep_df["x"]**2 + ep_df["y"]**2)

    ep_df['velocity'] = np.sqrt(ep_df["xdot"]**2 +ep_df["ydot"]**2).values

    for i in range(1,len(ep_df)):
        
        if ep_df.loc[ep_df.index[i],'position_error'] < POSITION_THRESHOLD and ep_df.loc[ep_df.index[i],'velocity'] < VELOCITY_THRESHOLD:
            
            ep_df.loc[ep_df.index[i],'concat_settle']= ep_df.loc[ep_df.index[i-1],'concat_settle']+1

        if ep_df.loc[ep_df.index[i-1],"T_settled"] == 1 or ep_df.loc[ep_df.index[i],"concat_settle"] == SETTLING_WINDOW:
            
            ep_df.loc[ep_df.index[i],"T_settled"] = 1

    if (ep_df["T_settled"] == 1).any():
        
        settles+=1
        settling_step.append(ep_df.loc[ep_df["T_settled"] == 1,"step"].iloc[0])

        if ((ep_df["concat_settle"] == 0) & (ep_df["T_settled"] == 1)).any():
        
            false_settles += 1

    df.loc[ep_df.index, ["concat_settle", "T_settled"]] = ep_df[["concat_settle", "T_settled"]]

settling_step= np.array(settling_step)
settled_timesteps = (df["T_settled"] == 1).sum()
total_timesteps =len(df)
        
# ------------------------
#       Summary
# ------------------------



print(f"Settled episodes: {settles}/{len(episode_lengths)}")

if settles > 0:

    print(f"False settles: "f"{100*false_settles/settles:.2f}%")

    print(f"Settled timesteps: " f"{100 * settled_timesteps / total_timesteps:.2f}%")

    print(f"Mean settling step: {settling_step.mean():.1f}")

    print(f"Median settling step: {np.median(settling_step):.1f}")

# --------------------
#       CDF
# --------------------

if settles > 0:

    x = np.arange(MAX_EPISODE_LENGTH)

    percentages = [100*np.mean(settling_step <= t)for t in x]

    plt.figure(figsize=(6,4))

    plt.step(x, percentages, where='post')

    plt.xlabel("Timestep")
    plt.ylabel("% episodes settled")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join('ball_on_plate/expert/mpc_horizons',f"settling_cdf_H_{H}.png"),dpi=300,)

    plt.close()