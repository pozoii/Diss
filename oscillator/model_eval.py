import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

from envs.harmonic_oscillator import HarmonicOscillatorEnv
from train import PolicyNet

# to retrieve trained models, run in the terminal: wandb artifact get <name of model> --root <path to save the model>
def load_policy(checkpoint_path="best_model.pt", device=None):

    if device is None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNet().to(device)

    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

def evaluate_controller(env,policy_fn,episodes=100,max_steps=500,):

    results = []

    for _ in tqdm(range(episodes), desc="Evaluating"):

        obs, info = env.reset()
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

        x, xdot = obs

        u = Kp * (target - x) - Kd * xdot

        return np.array([u], dtype=np.float32)

    return policy

def nn_policy(model, device):

    @torch.no_grad()
    def policy(obs, target):

        x, xdot = obs

        e = target - x

        inp = torch.tensor([[e, xdot]], dtype=torch.float32,device=device)

        u = model(inp).cpu().numpy()[0]

        return u.astype(np.float32)

    return policy
    
model, device = load_policy("training/best_model.pt")

env = HarmonicOscillatorEnv(render_mode=None)

pd_results = evaluate_controller(env, pd_policy(32, 2),episodes=1000)

nn_results = evaluate_controller(env,nn_policy(model, device),episodes=1000)

comparison = pd.DataFrame({"PD": pd_results.mean(),"NN": nn_results.mean(),})

print(comparison)