import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    return np.load(path)


def compute_physics_action(x, xdot, next_xdot, dt, m=1.0, k=10.0):

    xddot = (next_xdot - xdot) / dt

    return m * xddot + k * x


def analyze_dataset(npz_path, dt=0.01, m=1.0, k=10.0, sample=100000):

    data = load_npz(npz_path)

    state = data["state"]
    next_state = data["next_state"]
    action = data["action"]

    # extract variables
    x = state[:, 0]
    xdot = state[:, 1]

    next_xdot = next_state[:, 1]

    action = action.reshape(-1)


    # subsample
    N = len(x)

    idx = np.random.choice(N,min(sample, N),replace=False)

    x = x[idx]
    xdot = xdot[idx]
    next_xdot = next_xdot[idx]
    action = action[idx]


    # physics reconstruction
    u_phys = compute_physics_action(x,xdot,next_xdot,dt,m,k)


    # residual
    residual = u_phys - action


    # metrics
    mse = np.mean(residual**2)
    mae = np.mean(np.abs(residual))

    rel_error = (np.linalg.norm(residual)/(np.linalg.norm(action)+1e-8))

    corr = np.corrcoef(u_phys,action)[0,1]


    print("\n===== Physics Consistency Check =====")
    print(f"MSE residual: {mse:.6f}")
    print(f"MAE residual: {mae:.6f}")
    print(f"Relative error: {rel_error:.6f}")
    print(f"Correlation: {corr:.6f}")


    # -------------------------
    # plots
    # -------------------------

    plt.figure(figsize=(5,5))

    plt.scatter(action,u_phys,s=1,alpha=0.3)

    lims = [min(action.min(), u_phys.min()),max(action.max(), u_phys.max())]

    plt.plot(lims,lims,"r--")

    plt.xlabel("Expert action")
    plt.ylabel("m*xddot + k*x")
    plt.title("Physics consistency check")

    plt.grid(True)

    plt.show()



    plt.figure()

    plt.hist(residual,bins=100)

    plt.xlabel("Physics residual")
    plt.title("(m*xddot+k*x)-action")

    plt.show()


    plt.figure()

    plt.plot(action[:1000],label="action")

    plt.plot(u_phys[:1000],label="physics estimate")

    plt.legend()
    plt.title("First 1000 samples")

    plt.show()



if __name__ == "__main__":

    analyze_dataset("oscillator/data/train.npz",dt=0.01, m=1.0,k=10.0)