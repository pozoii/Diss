import numpy as np
from joblib import load
import time
import sys
import os
import pandas as pd
from ball_on_plate.expert.expert_MPC import  run_ExpertMPC

def main():
    Hs = [1,2,5,10,20,30,50,75,100,150,250,500]
    Hs= [100]
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0 

    H = Hs[idx]
    print(f"Running MPC with horizon H={H}")

    start_time = time.perf_counter()

    df = run_ExpertMPC(H=H,episodes=5,max_steps=5000)

    total_time = time.perf_counter() - start_time
    df["computation_time"] = total_time
    df["H"] = H

    
    os.makedirs("ball_on_plate/expert/mpc_horizons",exist_ok=True)
    df.to_csv(f"ball_on_plate/expert/mpc_horizons/H_{H}.csv",index=False)

if __name__ == "__main__":
    main()