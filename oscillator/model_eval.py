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

    return model, device

def evaluate_controller(env,policy_fn,init_configs,max_steps=500,):

    results = []

    for ini in tqdm(init_configs, desc="Evaluating"):

        obs, info = env.reset(options={"ini": ini})
        target = info["target"]

        errors = []
        actions = []

        for t in range(max_steps):

            action = policy_fn(obs, target)

            next_obs, reward, terminated, truncated, info = env.step(action)

            x = obs[0]
            error = target - x

            errors.append(error**2)
            actions.append(action[0]**2)

            obs = next_obs

            if terminated or truncated:
                break

        mse = np.mean(errors)
        control_cost = np.sum(actions)

        results.append({
            "mse": mse,
            "settling_time": t + 1,
            "control_cost": control_cost,
            "success": float(terminated),
        })

    return pd.DataFrame(results)

def pd_policy(Kp, Kd):
    
    @torch.no_grad()
    def policy(obs, target):

        x, xdot, xddot = obs

        u = Kp * (target - x) - Kd * xdot

        return np.array([u], dtype=np.float32)

    return policy

def nn_policy(model, device):

    @torch.no_grad()
    def policy(obs, target):

        x, xdot, xddot = obs

        e = target - x

        inp = torch.tensor([[e, xdot]], dtype=torch.float32,device=device)

        u = model(inp).cpu().numpy()[0]

        return u.astype(np.float32)

    return policy

all_results = []

model_dir = "oscillator/models"

env = HarmonicOscillatorEnv(render_mode=None)

inits = init_configs(10000,p_range=1, v_range=0.5)

pd_results = evaluate_controller(env, pd_policy(32, 2), init_configs= inits)

pd_results = {
            "model": "Expert PD",
            "mse": [pd_results["mse"].mean()],
            "settling_time": [pd_results["settling_time"].mean()],
            "control_cost": [pd_results["control_cost"].mean()],    
            "success": [pd_results["success"].mean()],
            }

print(f"\n===== Evaluation Results for Expert PD =====")
print(pd_results)

all_results.append(pd_results)

for model_path in glob.glob(os.path.join(model_dir, "*.pt")):

    model, device = load_policy(model_path)

    model_name = os.path.basename(model_path)

    env = HarmonicOscillatorEnv(render_mode=None)

    nn_results = evaluate_controller(env,nn_policy(model, device), init_configs= inits)

    nn_results = {
            "model": [model_name],
            "mse": [nn_results["mse"].mean()],
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