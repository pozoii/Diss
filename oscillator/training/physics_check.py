import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    data = np.load(path)
    return data


def compute_physics_action(x, xddot, m=1.0, k=10.0):
    x = x.reshape(-1)
    xddot = xddot.reshape(-1)
    return m * xddot + k * x


def analyze_dataset(npz_path, m=1.0, k=10.0, sample=100000):
    data = load_npz(npz_path)

    # adjust keys if needed
    obs = data["obs"]        # (N, 4) in your case
    action = data["action"]

    # extract signals
    x = obs[:, 0]
    xdot = obs[:, 1]
    xddot = obs[:, 2]
    target = obs[:, 3]

    action = action.reshape(-1)

    # optionally subsample for speed
    N = len(x)
    idx = np.random.choice(N, min(sample, N), replace=False)

    x = x[idx]
    xddot = xddot[idx]
    action = action[idx]

    # physics prediction
    u_phys = compute_physics_action(x, xddot, m=m, k=k)

    # residual
    residual = u_phys - action

    # ---- metrics ----
    mse = np.mean(residual**2)
    mae = np.mean(np.abs(residual))

    # normalized error (important!)
    rel_error = np.linalg.norm(residual) / (np.linalg.norm(action) + 1e-8)

    corr = np.corrcoef(u_phys, action)[0, 1]

    print("\n===== Physics Consistency Check =====")
    print(f"MSE residual: {mse:.6f}")
    print(f"MAE residual: {mae:.6f}")
    print(f"Relative error: {rel_error:.6f}")
    print(f"Correlation: {corr:.6f}")

    # ---- plots ----

    # scatter plot: physics vs action
    plt.figure()
    plt.scatter(action, u_phys, s=1, alpha=0.3)
    plt.xlabel("Expert action")
    plt.ylabel("m xddot + k x")
    plt.title("Physics consistency check")
    plt.plot([action.min(), action.max()],
             [action.min(), action.max()],
             'r--')
    plt.show()

    # residual histogram
    plt.figure()
    plt.hist(residual, bins=100)
    plt.title("Residual: (physics - action)")
    plt.show()

    # time series sanity check
    plt.figure()
    plt.plot(action[:1000], label="action")
    plt.plot(u_phys[:1000], label="physics estimate")
    plt.legend()
    plt.title("First 1000 samples")
    plt.show()


if __name__ == "__main__":
    analyze_dataset("training/train.npz", m=1.0, k=10.0)
    