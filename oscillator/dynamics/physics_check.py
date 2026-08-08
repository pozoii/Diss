import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "oscillator/data/physics_check_data/physics_check.npz"

PLOT_DIR = "oscillator/dynamics/physics_check_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

DT = 0.01
MASS = 1.0
K = 10.0

HORIZON = 92
N_ROLLOUTS = 2


# ============================================================
# Dataset
# ============================================================

def load_dataset(path):

    data = np.load(path)

    states = data["state"]
    next_states = data["next_state"]
    actions = data["actions"].reshape(-1)
    dones = data["dones"]

    return states, next_states, actions, dones


def get_episodes(states, next_states, actions, dones):

    episodes = {}

    start = 0
    episode_id = 0

    for i, done in enumerate(dones):

        if done:

            end = i + 1

            episodes[episode_id] = {"states": states[start:end],"next_states": next_states[start:end],"actions": actions[start:end],}

            episode_id += 1
            start = end

    return episodes


# ============================================================
# Forward dynamics model
# ============================================================

def predict_next_state(state, action):

    x = state[0]
    xdot = state[1]

    xddot = (action - K * x) / MASS

    next_xdot = xdot + DT * xddot
    next_x = x + DT * next_xdot

    return np.array([next_x, next_xdot],dtype=np.float32 )


# ============================================================
# One-step validation
# ============================================================

def one_step_validation(episodes):

    true_states = []
    predicted_states = []

    for episode in episodes.values():

        states = episode["states"]
        actions = episode["actions"]

        for i in range(len(states)):

            prediction = predict_next_state(states[i],actions[i])

            predicted_states.append(prediction)

            true_states.append(episode["next_states"][i])

    true_states = np.asarray(true_states)
    predicted_states = np.asarray(predicted_states)

    errors = predicted_states - true_states

    metrics = {"x_rmse": np.sqrt(np.mean(errors[:, 0] ** 2)),"x_mae": np.mean(np.abs(errors[:, 0])),"xdot_rmse": np.sqrt(np.mean(errors[:, 1] ** 2)),"xdot_mae": np.mean(np.abs(errors[:, 1])),}

    return true_states, predicted_states, metrics


def save_metrics(one_step_metrics, rollout_rmse_x, rollout_rmse_xdot):

    metrics = {"one_step_x_rmse": one_step_metrics["x_rmse"],"one_step_x_mae": one_step_metrics["x_mae"],"one_step_xdot_rmse": one_step_metrics["xdot_rmse"],"one_step_xdot_mae": one_step_metrics["xdot_mae"],"rollout_50_x_rmse": rollout_rmse_x,"rollout_50_xdot_rmse": rollout_rmse_xdot,}

    output_path = os.path.join(PLOT_DIR,"validation_metrics.csv")

    with open(output_path, "w") as f:

        f.write("metric,value\n")

        for name, value in metrics.items():

            f.write(f"{name},{value:.10e}\n")

    print(f"Metrics saved to {output_path}")

# ============================================================
# One-step plot
# ============================================================

def plot_one_step_predictions(true_states, predicted_states):

    fig, axes = plt.subplots(1,2,figsize=(10, 4))

    axes[0].scatter(true_states[:, 0],predicted_states[:, 0],s=2,alpha=0.2)

    limits = [min(true_states[:, 0].min(), predicted_states[:, 0].min()),max(true_states[:, 0].max(), predicted_states[:, 0].max())]

    axes[0].plot(limits,limits,linestyle="--")

    axes[0].set_xlabel("MuJoCo $x_{t+1}$ (m)")
    axes[0].set_ylabel("Model $\\hat{x}_{t+1}$ (m)")
    axes[0].set_title("Position")

    axes[1].scatter(true_states[:, 1],predicted_states[:, 1],s=2,alpha=0.2)

    limits = [ min(true_states[:, 1].min(), predicted_states[:, 1].min()),max(true_states[:, 1].max(), predicted_states[:, 1].max())]

    axes[1].plot(limits,limits,linestyle="--")

    axes[1].set_xlabel("MuJoCo $\\dot{x}_{t+1}$ (m/s)")
    axes[1].set_ylabel("Model $\\hat{\\dot{x}}_{t+1}$ (m/s)")
    axes[1].set_title("Velocity")

    plt.tight_layout()

    plt.savefig(os.path.join(    PLOT_DIR,    "one_step_prediction.png"),dpi=300,bbox_inches="tight")

    plt.close()


# ============================================================
# Long-horizon validation
# ============================================================

def rollout_validation(episodes,horizon=50,n_rollouts=10):

    candidates = [episode_id for episode_id, episode in episodes.items() if len(episode["states"]) >= horizon + 1]

    if len(candidates) < n_rollouts:

        raise ValueError(f"Only {len(candidates)} episodes are long enough "f"for a {horizon}-step rollout.")

    selected = np.random.choice(candidates,size=n_rollouts,replace=False)

    rollout_errors = []
    representative = None

    for episode_id in selected:

        episode = episodes[episode_id]

        states = episode["states"]
        actions = episode["actions"]

        max_start = len(states) - horizon - 1

        start = np.random.randint(0,max_start + 1)

        true_states = states[start:start + horizon + 1]

        rollout_actions = actions[start:start + horizon]

        predicted_states = np.zeros_like(true_states)

        predicted_states[0] = true_states[0]

        for t in range(horizon):

            predicted_states[t + 1] = predict_next_state(predicted_states[t],rollout_actions[t] )

        errors = predicted_states - true_states

        rollout_errors.append(errors)

        if representative is None:

            representative = (true_states,predicted_states)

    rollout_errors = np.asarray(rollout_errors)

    rmse_x = np.sqrt(np.mean(rollout_errors[:, 1:, 0] ** 2))

    rmse_xdot = np.sqrt(np.mean(rollout_errors[:, 1:, 1] ** 2))

    return (rmse_x,rmse_xdot,representative )


# ============================================================
# Representative rollout plot
# ============================================================

def plot_representative_rollout(true_states,predicted_states):

    t = np.arange(len(true_states)) * DT

    fig, axes = plt.subplots(2,1,figsize=(8, 6),sharex=True)

    axes[0].plot(t,true_states[:, 0],label="MuJoCo",linewidth=2)

    axes[0].plot(t,predicted_states[:, 0],label="Forward model",linewidth=2)

    axes[0].set_ylabel("$x$ (m)")
    axes[0].set_title("50-step forward model rollout")

    axes[0].legend()

    axes[1].plot(t,true_states[:, 1],label="MuJoCo",linewidth=2)

    axes[1].plot(t,predicted_states[:, 1],label="Forward model",linewidth=2)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("$\\dot{x}$ (m/s)")

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(os.path.join(    PLOT_DIR,    "50_step_rollout.png"),dpi=300,bbox_inches="tight")

    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading dataset...")

    states, next_states, actions, dones = load_dataset(DATA_PATH)

    print(f"Transitions: {len(states):,}")

    episodes = get_episodes(states,next_states,actions,dones)

    print(f"Episodes: {len(episodes)}")

    # --------------------------------------------------------
    # One-step validation
    # --------------------------------------------------------

    print("\n===== One-Step Validation =====")

    true_states, predicted_states, metrics = ( one_step_validation(episodes))

    print( f"Position RMSE: {metrics['x_rmse']:.8f} m")

    print(f"Position MAE:  {metrics['x_mae']:.8f} m")

    print(f"Velocity RMSE: {metrics['xdot_rmse']:.8f} m/s" )

    print(f"Velocity MAE:  {metrics['xdot_mae']:.8f} m/s")

    plot_one_step_predictions(true_states, predicted_states)

    # --------------------------------------------------------
    # Long-horizon validation
    # --------------------------------------------------------

    print("\n===== 50-Step Rollout Validation =====")

    rmse_x, rmse_xdot, representative = (rollout_validation(episodes,horizon=HORIZON,n_rollouts=N_ROLLOUTS) )

    print( f"Position RMSE: {rmse_x:.8f} m")

    print( f"Velocity RMSE: {rmse_xdot:.8f} m/s")

    plot_representative_rollout(representative[0],representative[1])

    save_metrics(metrics,rmse_x,rmse_xdot)

    print("\nValidation complete.")