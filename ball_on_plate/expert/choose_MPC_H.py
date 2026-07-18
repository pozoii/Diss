import numpy as np
from ball_on_plate.env.ball_on_plate import BallOnPlateEnv
from ball_on_plate.expert.expert_MPC import ExpertMPC, run_ExpertMPC
import pandas as pd
from joblib import load

import sys
import os
import pandas as pd

from ball_on_plate.expert.expert_MPC import run_ExpertMPC



Hs = [1,2,5,10,20,30,50,75,100,150,250,500]


def main():

    idx = int(sys.argv[1])
    H = Hs[idx]
    print(f"Running MPC with horizon H={H}")

    # Collect data
    df = run_ExpertMPC(H=H,episodes=25,max_steps=5000)

    # Save raw trajectories
    os.makedirs("ball_on_plate/expert/mpc_horizons",exist_ok=True)

    df.to_csv(f"ball_on_plate/expert/mpc_horizons/H_{H}.csv",index=False)

    metrics = analyze_performance_and_time(df)
    metrics["H"] = H

    pd.DataFrame([metrics]).to_csv(f"ball_on_plate/results/mpc_horizons/metrics_H_{H}.csv",index=False)

