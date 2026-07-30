import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
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

# For expert:
parser.add_argument(
    "--job-id",
    type=int,
    default=0
)

parser.add_argument(
    "--episode-offset",
    type=int,
    default=0
)

args = parser.parse_args()

# ============================================================
# Configuration
# ============================================================
MAX_STEPS = 5000

RESULTS_DIR = "ball_on_plate/results"

SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")
TRAJECTORY_DIR = os.path.join(RESULTS_DIR, "trajectories")
EPISODE_DIR = os.path.join(RESULTS_DIR, "episode_metrics")

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(TRAJECTORY_DIR, exist_ok=True)
os.makedirs(EPISODE_DIR, exist_ok=True)


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
    state_std_dev= state_std.to(device)
    action_mean_dev = action_mean.to(device)
    action_std_dev = action_std.to(device)

    @torch.no_grad()
    def policy(obs):

        x = torch.tensor(obs,dtype=torch.float32,device=device)


        x_norm = (x - state_mean_dev) / state_std_dev
        action_norm = model(x_norm)

        action = torch.clamp(action_norm *action_std_dev+action_mean_dev,-10,10)

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

def evaluate_controller(env,controller,n_episodes,model_name,difficulty_name,job_id,episode_offset,expert=None):

    episode_results = []

    trajectory_buffer = []
    episode_buffer = []

    trajectory_path = os.path.join(TRAJECTORY_DIR,f"{model_name}_{difficulty_name}_job{job_id:03d}.csv")

    episode_path = os.path.join(EPISODE_DIR,f"{model_name}_{difficulty_name}_job{job_id:03d}.csv")

    for ep in tqdm(range(n_episodes),desc="Evaluating"):

        obs,info = env.reset()

        if expert is not None:
            expert.reset()

        global_episode = episode_offset + ep
        mse=[]
        control_cost=[]

        terminated=False
        truncated=False

        for t in range(MAX_STEPS):

            state = obs.copy()

            action = controller(obs)

            next_obs,reward,terminated,truncated,info = env.step(action)

            error = np.linalg.norm(obs[:2])

            mse.append(error**2)

            control_cost.append(np.sum(action**2))

            trajectory_buffer.append({

                "job_id": job_id,
                "episode": global_episode,
                "step": t,

                # Current state
                "x": state[0],
                "y": state[1],
                "xdot": state[2],
                "ydot": state[3],
                "alpha": state[4],
                "beta": state[5],
                "alphadot": state[6],
                "betadot": state[7],

                # Action
                "roll_torque": action[0],
                "pitch_torque": action[1],

                # Next state
                "x_next": next_obs[0],
                "y_next": next_obs[1],
                "xdot_next": next_obs[2],
                "ydot_next": next_obs[3],
                "alpha_next": next_obs[4],
                "beta_next": next_obs[5],
                "alphadot_next": next_obs[6],
                "betadot_next": next_obs[7],

                # Environment information
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,

                "ball_stable": info["ball_stable"],
                "ball_lost": info["ball_lost"],
            })

            obs = next_obs

            if terminated or truncated:
                break

        episode_result = {

            "job_id": job_id,
            "episode": global_episode,

            "success": float(info["ball_stable"]),
            "failure": float(info["ball_lost"]),

            "steps": t + 1,

            "settling_time": (
                t + 1
                if info["ball_stable"]
                else np.nan
            ),

            "mse": np.mean(mse),

            "control_cost": np.sum(control_cost),

            "final_position_error": np.linalg.norm(obs[:2]),
        }

        episode_results.append(episode_result)

        episode_buffer.append(episode_result)


        # Save every 5 episodes
        if (ep + 1) % 5 == 0:

            if trajectory_buffer:
                trajectory_df = pd.DataFrame(trajectory_buffer)

                trajectory_df.to_csv(
                    trajectory_path,
                    mode="a",
                    header=not os.path.exists(trajectory_path),
                    index=False,
                )

                trajectory_buffer.clear()


            if episode_buffer:
                episode_df = pd.DataFrame(episode_buffer)

                episode_df.to_csv(
                    episode_path,
                    mode="a",
                    header=not os.path.exists(episode_path),
                    index=False,
                )

                episode_buffer.clear()

    if trajectory_buffer:

        trajectory_df = pd.DataFrame(trajectory_buffer)

        trajectory_df.to_csv(
            trajectory_path,
            mode="a",
            header=not os.path.exists(trajectory_path),
            index=False,
        )


    if episode_buffer:

        episode_df = pd.DataFrame(episode_buffer)

        episode_df.to_csv(
            episode_path,
            mode="a",
            header=not os.path.exists(episode_path),
            index=False,
        )

    return pd.DataFrame(episode_results)

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

nn_results = evaluate_controller(env,controller,args.episodes,model_name=model_name, difficulty_name=difficulty_name, job_id=args.job_id, episode_offset= args.episode_offset, expert=expert)

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

output = os.path.join(SUMMARY_DIR,f"{model_name}_{difficulty_name}_job{args.job_id:03d}.csv")

df.to_csv(output,index=False)

print("\nSaved:")
print(output)