import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
import glob
from oscillator.envs.harmonic_oscillator import HarmonicOscillatorEnv
from oscillator.train import PolicyNet
from datetime import datetime

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_")

def init_configs(n_eps,p_range=1, v_range=0.5, seed=42):
    rng = np.random.default_rng(seed)

    inits = []
    for _ in range(n_eps):
        inits.append({
            "pos": rng.uniform(-p_range, p_range),
            "vel": rng.uniform(-v_range, v_range),
        })

    return inits


# to retrieve trained models, run in the terminal: wandb artifact get <name of model> --root <path to save the model>
def load_policy(checkpoint_path="best_model.pt", device=None):

    if device is None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNet().to(device)

    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm = np.load("oscillator/data/normalization.npz")

    state_mean = torch.tensor(
        norm["state_mean"],
        dtype=torch.float32,
        device=device
    )

    state_std = torch.tensor(
        norm["state_std"],
        dtype=torch.float32,
        device=device
    )

    action_mean = torch.tensor(
        norm["action_mean"],
        dtype=torch.float32,
        device=device
    )

    action_std = torch.tensor(
        norm["action_std"],
        dtype=torch.float32,
        device=device
    )

    return model, device, state_mean, state_std, action_mean, action_std


def evaluate_controller(env,policy_fn,init_configs,max_steps=500,):

    results = []

    for ini in tqdm(init_configs, desc="Evaluating"):

        obs, info = env.reset(options={"ini": ini})

        pos_errors = []
        vel_errors = []
        actions = []

        for t in range(max_steps):

            action = policy_fn(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            x = obs[0]
            xdot=obs[1]
           

            pos_errors.append(abs(x))
            vel_errors.append(abs(xdot))
            actions.append(action[0]**2)

            obs = next_obs

            if terminated or truncated:
                break

        cumulative_reward = np.sum(-10*np.array(pos_errors)-1*np.array(vel_errors)-0.01*np.array(actions))

        results.append({
            "cum_reward": cumulative_reward,
            "settling_time": t + 1,
            "control_cost": np.sum(actions),
            "success": float(terminated),
        })

    return pd.DataFrame(results)

def pd_policy(Kp, Kd):
    
    @torch.no_grad()
    def policy(obs):

        x, xdot = obs

        u = -Kp *  x - Kd * xdot

        return np.array([u], dtype=np.float32)

    return policy

def nn_policy(model, device, state_mean, state_std, action_mean, action_std):

    @torch.no_grad()
    def policy(obs):

        x, xdot = obs

        inp = torch.tensor([[x, xdot]], dtype=torch.float32,device=device)

        inp = (inp - state_mean) / state_std

        pred_norm = model(inp)

        u = pred_norm * action_std + action_mean


        return u.cpu().numpy().reshape(-1).astype(np.float32)

    return policy

all_results = []

model_dir = "oscillator/models"

env = HarmonicOscillatorEnv(render_mode=None)

inits = init_configs(10000,p_range=1, v_range=0.5)

pd_results = evaluate_controller(env, pd_policy(16, 8), init_configs= inits)

pd_results = {
            "model": "Expert PD",
            "cum_reward": [pd_results["cum_reward"].mean()],
            "settling_time": [pd_results["settling_time"].mean()],
            "control_cost": [pd_results["control_cost"].mean()],    
            "success": [pd_results["success"].mean()],
            }

print(f"\n===== Evaluation Results for Expert PD =====")
print(pd_results)

all_results.append(pd_results)

for model_path in glob.glob(os.path.join(model_dir, "*.pt")):

    model, device, state_mean, state_std, action_mean, action_std = load_policy(model_path)

    model_name = os.path.basename(model_path)

    env = HarmonicOscillatorEnv(render_mode=None)

    nn_results = evaluate_controller(env,nn_policy(model, device, state_mean,state_std,action_mean,action_std), init_configs= inits)

    nn_results = {
            "model": [model_name],
            "cum_reward": [nn_results["cum_reward"].mean()],
            "settling_time": [nn_results["settling_time"].mean()],
            "control_cost": [nn_results["control_cost"].mean()],
            "success": [nn_results["success"].mean()],
        }

    print(f"\n===== Evaluation Results for {model_name} =====")
    print(nn_results)

    all_results.append(nn_results)

df = pd.DataFrame(all_results)
output_path = f"oscillator/results/eval_results_{timestamp}.csv"
df.to_csv(output_path, index=False)

print(output_path)