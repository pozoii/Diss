import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from datetime import datetime
from joblib import load

from ball_on_plate.envs.ball_on_plate import BallOnPlateEnv
from ball_on_plate.train import PolicyNet
from ball_on_plate.expert.expert_MPC import ExpertMPC

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--difficulty",
    type=str,
    required=True
    )

parser.add_argument(
    "--model",
    type=str,
    required=True
    )

parser.add_argument(
    "--episodes",
    type=int,
    default=1000
    )

args = parser.parse_args()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_")


# ============================================================
# Configuration
# ============================================================
MAX_STEPS = 5000

MODEL_DIR = "ball_on_plate/models"
OUTPUT_DIR = "ball_on_plate/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


DIFFICULTIES = {
    "easy":{"max_pos_reset":0.15,"max_vel_reset":0.5},
    "medium":{"max_pos_reset":0.20,"max_vel_reset":0.5},
    "hard":{"max_pos_reset":0.20,"max_vel_reset":0.8},
    "very_hard":{"max_pos_reset":0.24,"max_vel_reset":1.0}
    }

# ============================================================
# Normalization
# ============================================================

norm = np.load("ball_on_plate/data/normalization.npz")

state_mean = torch.tensor(norm["state_mean"],dtype=torch.float32)
state_std = torch.tensor(norm["state_std"],dtype=torch.float32)

action_mean = torch.tensor(norm["action_mean"],dtype=torch.float32)

action_std = torch.tensor(norm["action_std"],dtype=torch.float32)

# ============================================================
# Model loading
# ============================================================

def load_policy(path):

    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

    model = PolicyNet().to(device)

    checkpoint = torch.load(path,map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model, device

# ============================================================
# NN controller
# ============================================================

def nn_policy(model,device):

    state_mean_dev = state_mean.to(device)
    state_std_dev = state_std.to(device)
    action_mean_dev = action_mean.to(device)
    action_std_dev = action_std.to(device)

    @torch.no_grad()
    def policy(obs):

        x = torch.tensor(obs,dtype=torch.float32,device=device)


        x_norm = (x - state_mean) / state_std
        action_norm = model(x_norm)

        action = torch.clamp(action_norm *action_std+action_mean,-10,10)

        return action.cpu().numpy()


    return policy

# ============================================================
# Expert controller
# ============================================================

def create_expert(env):

    plate_models = load("ball_on_plate/dynamics/plate_models.joblib")

    plate_models = (plate_models["roll_model"],plate_models["pitch_model"])

    expert = ExpertMPC(env=env, plate_models=plate_models,H=50)

    return expert

# ============================================================
# Evaluation
# ============================================================

def evaluate_controller(env,controller,n_episodes):

    results=[]

    for ep in tqdm(range(n_episodes),desc="Evaluating"):

        obs,info = env.reset()

        if expert is not None:
            expert.reset()

        mse=[]
        control_cost=[]

        terminated=False
        truncated=False

        for t in range(MAX_STEPS):

            action = controller(obs)

            next_obs,reward,terminated,truncated,info = env.step(action)

            error = np.linalg.norm(obs[:2])

            mse.append(error**2)

            control_cost.append(np.sum(action**2))

            obs = next_obs

            if terminated or truncated:
                break

        results.append({

            "success": float(info["ball_stable"]),

            "failure": float(info["ball_lost"]),

            "steps": t+1,

            "settling_time": t+1 if info["ball_stable"]else np.nan,

            "mse": np.mean(mse),

            "control_cost":np.sum(control_cost),

            "final_position_error": np.linalg.norm(obs[:2])
            })


    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================


all_results=[]

difficulty_name = args.difficulty

difficulty = DIFFICULTIES[difficulty_name]



print("\n")
print("="*70)
print(f"Difficulty: {difficulty_name}")
print("="*70)

env = BallOnPlateEnv(render_mode=None, max_pos_reset=difficulty["max_pos_reset"],max_vel_reset=difficulty["max_vel_reset"],settling_window=50)

expert = None

if args.model == "expert":
    expert = create_expert(env)

    def controller(obs):
        return expert.control(obs)
    
    model_name = "Expert_MPC"
    
else:

    checkpoint = (f"ball_on_plate/models/{args.model}.pt")

    model,device = load_policy(checkpoint)

    controller = nn_policy(model,device)

    model_name = args.model

nn_results = evaluate_controller(env,controller,args.episodes)

summary={

    "model": model_name,

    "difficulty":difficulty_name,

    "success": nn_results["success"].mean(),

    "failure": nn_results["failure"].mean(),

    "settling_time": nn_results["settling_time"].mean(),

    "mse": nn_results["mse"].mean(),

    "control_cost": nn_results["control_cost"].mean(),

    "final_error": nn_results["final_position_error"].mean()
    }

print(summary)

all_results.append(summary)

env.close()

# ============================================================
# Save
# ============================================================

df = pd.DataFrame(all_results)


output = (f"ball_on_plate/results/{model_name}_{difficulty_name}.csv")

df.to_csv(output,index=False)

print("\nSaved:")
print(output)