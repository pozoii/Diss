import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ball_on_plate.data.bop_dataset import BoPDataset
from ball_on_plate.dynamics.BoP_forward_dynamics import BallOnPlateDynamics
from torch.utils.data import DataLoader

import wandb
import os
from datetime import datetime
import argparse
from joblib import load

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_")
job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
os.environ["WANDB_MODE"] = "offline"

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,2)
        )

    def forward(self, x):
        return self.net(x)

class EarlyStopping:

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):

        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False  # no stop

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                return True  # stop

        return False
    
def pinn_loss(pred_action,true_action, state, next_state, dynamics, state_std, action_std, lambd=1.0 ):
 
    # Behavioral cloning loss
    bc_loss = torch.mean(((pred_action - true_action) / action_std)**2)

    pred_next_state = dynamics(state, pred_action, )

    physics_loss = torch.mean(((pred_next_state-next_state)/state_std)**2)

    total_loss = (1-lambd) * bc_loss + lambd * physics_loss

    return total_loss, bc_loss, physics_loss

def train(model, train_loader, val_loader, lambd, dynamics, epochs=20, lr=1e-3):

    wandb.init(project="diss-ball_on_plate",
               config={"lr": lr,"lambda": lambd,"batch_size": 64},
               name=f"{timestamp}_lambda_{lambd}_job{job_id}",
               mode='offline')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = np.load("ball_on_plate/data/normalization.npz")
        
    state_mean = torch.tensor(norm["state_mean"], dtype=torch.float32).to(device)
    state_std = torch.tensor(norm["state_std"], dtype=torch.float32).to(device)

    action_mean = torch.tensor(norm["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(norm["action_std"], dtype=torch.float32).to(device)

    model.to(device)
    dynamics.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

    history = []
    best_val_loss = float("inf")


    for epoch in range(epochs):
        model.train()

        train_loss = {
                    'train_loss' : 0.0,
                    'train_bc_loss' : 0.0,
                    'train_physics_loss' : 0.0,
                    'train_physics_ratio' : 0.0
                    }

        for i, batch in enumerate(train_loader):

            state = batch["state"].to(device,non_blocking=True)
            next_state = batch["next_state"].to(device,non_blocking=True)
            action = batch["action"].to(device,non_blocking=True)   

            state_norm = (state - state_mean) / state_std

            pred_norm = model(state_norm)
            pred_action = torch.clamp(pred_norm * action_std + action_mean, -10,10)
            


            loss, bc_loss, physics_loss = pinn_loss(pred_action=pred, true_action=action, state=state, next_state=next_state, lambd=lambd, dynamics=dynamics, state_std=state_std,action_std=action_std)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss['train_loss'] += loss.item()
            train_loss['train_bc_loss'] += bc_loss.item()
            train_loss['train_physics_loss'] += physics_loss.item()
            

            if i % 10000 == 0:
                print(
                    f"Epoch {epoch+1} "
                    f"Batch {i}/{len(train_loader)} "
                    f"Loss {loss.item():.4e}"
                    )
        
        for k in train_loss:
            train_loss[k] /= len(train_loader)

        train_loss["train_physics_ratio"] = train_loss["train_physics_loss"] / (train_loss["train_bc_loss"] + 1e-8)

        wandb.log({
            "train/loss": train_loss["train_loss"],
            "train/bc_loss": train_loss["train_bc_loss"],
            "train/physics_loss": train_loss["train_physics_loss"],
            "train/physics_ratio": train_loss["train_physics_ratio"],
            "epoch": epoch
            })

        model.eval()
        val_loss = {'val_loss':0.0,
                    'val_bc_loss':0.0,
                    'val_physics_loss':0.0,
                    'val_physics_ratio':0.0,
                    }

        with torch.no_grad():
            for batch in val_loader:

                state = batch["state"].to(device,non_blocking=True)
                next_state = batch["next_state"].to(device,non_blocking=True)
                action = batch["action"].to(device,non_blocking=True)

                state_norm = (state - state_mean) / state_std

                pred_norm = model(state_norm)

                pred_action = torch.clamp(pred_norm * action_std + action_mean,-10,10)

                loss, bc_loss, physics_loss = pinn_loss(pred_action=pred_action, true_action=action, state = state, next_state=next_state, lambd = lambd, dynamics=dynamics)
                

                val_loss['val_loss'] += loss.item()
                val_loss['val_bc_loss'] += bc_loss.item()
                val_loss['val_physics_loss'] += physics_loss.item()
                


        for k in val_loss:
            val_loss[k] /= len(val_loader)
        
        val_loss["val_physics_ratio"] = val_loss["val_physics_loss"] / (val_loss["val_bc_loss"] + 1e-8)

        wandb.log({
            "val/loss": val_loss["val_loss"],
            "val/bc_loss": val_loss["val_bc_loss"],
            "val/physics_loss": val_loss["val_physics_loss"],
            "val/physics_ratio": val_loss["val_physics_ratio"],
            "epoch": epoch
            })

        history.append({
                "epoch": epoch + 1,

                "train_loss": train_loss["train_loss"],
                "train_bc_loss": train_loss["train_bc_loss"],
                "train_physics_loss": train_loss["train_physics_loss"],

                "val_loss": val_loss["val_loss"],
                "val_bc_loss": val_loss["val_bc_loss"],
                "val_physics_loss": val_loss["val_physics_loss"],
                "val_physics_ratio": val_loss["val_physics_ratio"]
            })

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss['train_loss']:.6f} "
            f"Val: {val_loss['val_loss']:.6f} "
            )

        if val_loss["val_loss"] < best_val_loss:

            best_val_loss = val_loss["val_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch+1,
                "val_loss": val_loss,
                "lambda":lambd,
                "physics_dt": dt,
                "plate_joint_limit": plate_joint_limit,
                "roll_coef": roll_model.coef_,
                "roll_intercept": roll_model.intercept_,
                "pitch_coef": pitch_model.coef_,
                "pitch_intercept": pitch_model.intercept_,
            }, f"ball_on_plate/models/{timestamp}_lambda={lambd}.pt")

        stop = early_stopper.step(val_loss["val_loss"])

        if stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    artifact = wandb.Artifact(f"model_lambda_{lambd}",type="model")
    artifact.add_file(f"ball_on_plate/models/{timestamp}_lambda={lambd}.pt")
    wandb.log_artifact(artifact)

    df = pd.DataFrame(history)
    df.to_csv(f"ball_on_plate/training_logs/training_log_{timestamp}_lambda={lambd}.csv", index=False)
    wandb.finish()
    

if __name__ == "__main__":

    train_ds = BoPDataset("ball_on_plate/data/train.npz")
    val_ds   = BoPDataset("ball_on_plate/data/val.npz")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2,pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = PolicyNet()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_", type=float)
    args = parser.parse_args()

    plate_models = load("ball_on_plate/dynamics/plate_models.joblib")
    roll_model, pitch_model = plate_models["roll_model"],plate_models["pitch_model"]

    plate_joint_limit = np.deg2rad(30)
    dt = 0.02


    lambd = args.lambda_
    dynamics= BallOnPlateDynamics(dt,roll_model,pitch_model,plate_joint_limit)
    train(model, train_loader, val_loader, lambd=lambd, dynamics=dynamics)
