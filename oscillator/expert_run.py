import numpy as np
from envs.harmonic_oscillator import HarmonicOscillatorEnv
import pandas as pd

def PD(Kp=20,Kd=5,episodes=50):

    env = HarmonicOscillatorEnv(render_mode=None)
    results=[]
    for ep in range(episodes):
        obs, info = env.reset()
        target = info["target"]

        episode_reward = 0.0
        episode_steps = 0

        terminated , truncated = False, False

        xs = []
        xdots = []
        errors = []
        actions = []
   

        while not terminated and not truncated:

            x, xdot = obs

            action = np.array([Kp * (target - x) - Kd * xdot], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            

            # --- logging ---
            episode_reward += reward
            episode_steps += 1


            xs.append(x)
            xdots.append(xdot)

            error = x - target
            errors.append(error)
            actions.append(action[0])

        xs = np.array(xs)
        xdots = np.array(xdots)
        errors = np.array(errors)
        actions = np.array(actions)

        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        control_cost = np.mean(actions**2)
        M_error = np.max(np.abs(errors))
        final_error = np.abs(errors[-1])

        results.append({
            "Kp": Kp,
            "Kd": Kd,
            "episode": ep,
            "reward": episode_reward,
            "mse": mse,
            "mae": mae,
            "control_cost": control_cost,
            "Max_error": M_error,
            "final_error": final_error,
            "settling_time": episode_steps,
            "terminated": terminated,
            "truncated": truncated,
            "target": target,
            "final_x": obs[0],
            "final_xdot": obs[1],
        })

    env.close()
    return pd.DataFrame(results)


if __name__ == "__main__":
    all_dfs=[]
    Kps=[0,0.1,0.5,1,2,4,8,16,32,64]
    Kds=[0,0.1,0.5,1,2,4,8,16,32,64]
    results={}
    for Kp in Kps:
        for Kd in Kds:
            print(f"Testing Kp={Kp}, Kd={Kd}")
            results= PD(Kp,Kd,episodes=50)
            all_dfs.append(results)

    DF = pd.concat(all_dfs)
    DF.to_csv("PD_sweep.csv", index=False)
            
