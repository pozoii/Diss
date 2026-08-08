import numpy as np
import pandas as pd

from oscillator.envs.harmonic_oscillator import HarmonicOscillatorEnv


def PD(Kp=16, Kd=8, episodes=50):

    env = HarmonicOscillatorEnv(render_mode=None)

    results = []

    for ep in range(episodes):

        obs, info = env.reset()
        target = info["target"]

        terminated = False
        truncated = False

        while not terminated and not truncated:

            x, xdot = obs

            action = np.array(
                [Kp * (target - x) - Kd * xdot],
                dtype=np.float32,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)

            results.append(
                {
                    "Kp": Kp,
                    "Kd": Kd,
                    "episode": ep,
                    "target": target,
                    "x": x,
                    "xdot": xdot,
                    "action": action[0],
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

            obs = next_obs

    env.close()

    return pd.DataFrame(results)


if __name__ == "__main__":

    all_dfs = []

    Kps = [0, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64]
    Kds = [0, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64]

    for Kp in Kps:
        for Kd in Kds:
            print(f"Testing Kp={Kp}, Kd={Kd}")
            df = PD(Kp, Kd, episodes=50)
            all_dfs.append(df)

    DF = pd.concat(all_dfs, ignore_index=True)
    DF.to_csv("oscillator/expert/PD_sweep.csv", index=False)
