import numpy as np
from joblib import load
import time
import sys
import os
import pandas as pd
from ball_on_plate.expert.expert_MPC import  run_ExpertMPC

def analyze_performance_and_time(df, target=np.array([0.0,0.0]),settling_threshold=0.02, dt=0.002):

    metrics = {}

    pos_error = np.sqrt((df["x"] - target[0])**2 +(df["y"] - target[1])**2)

    metrics["mean_position_error"] = pos_error.mean()
    metrics["max_position_error"] = pos_error.max()

    torque = np.sqrt(df["roll_torque"]**2 + df["pitch_torque"]**2)

    metrics["mean_torque"] = torque.mean()
    metrics["control_cost"] = np.sum(torque**2)

    episodes = df["episode"].unique()

    episode_lengths = []
    failures = []

    settling_times = []


    for ep in episodes:

        ep_df = df[df["episode"] == ep]

        episode_lengths.append(len(ep_df))


        # Did it fail?
        failures.append(
            ep_df["done"].iloc[-1]
        )

        error = np.sqrt((ep_df["x"] - target[0])**2 +(ep_df["y"] - target[1])**2).values

        settled = np.where(error < settling_threshold)[0]

        settling_window = 100

        settled = np.where(error < settling_threshold)[0]

        time = np.nan

        for idx in settled:
            if idx + settling_window < len(error):
                if np.all(error[idx:idx+settling_window] < settling_threshold):
                    time = idx * dt
                    break

        settling_times.append(time)

    metrics["mean_episode_length"] = np.mean(episode_lengths)

    metrics["failure_rate"] = np.mean(failures)


    metrics["mean_settling_time"] = np.nanmean(settling_times)


    metrics["computation_time_per1000"] = df["computation_time"].iloc[0]/len(df)*1000

    return metrics

def main():
    Hs = [1,2,5,10,20,30,50,75,100,150,250,500]
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0 

    H = Hs[idx]
    print(f"Running MPC with horizon H={H}")

    start_time = time.perf_counter()

    df = run_ExpertMPC(H=H,episodes=25,max_steps=5000)

    total_time = time.perf_counter() - start_time
    df["computation_time"] = total_time
    df["H"] = H

    
    os.makedirs("ball_on_plate/expert/mpc_horizons",exist_ok=True)
    df.to_csv(f"ball_on_plate/expert/mpc_horizons/H_{H}.csv",index=False)

    metrics = analyze_performance_and_time(df)
    metrics["H"] = H

    pd.DataFrame([metrics]).to_csv(f"ball_on_plate/expert/mpc_horizons/metrics_H_{H}.csv",index=False)

if __name__ == "__main__":
    main()