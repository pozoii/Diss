import re
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "oscillator/results/.csv"
df = pd.read_csv(csv_file)

def extract_lambda(name):
    if name == "Expert PD":
        return None

    match = re.search(r"lambda[_=](\d+(?:\.\d+)?)", name)
    if not match:
        raise ValueError(f"No lambda found in: {name}")

    return float(match.group(1))

df["lambda"] = df["model"].apply(extract_lambda)

baseline = df[df["model"] == "Expert PD"].iloc[0]
models = df[df["model"] != "Expert PD"].sort_values("lambda")


def plot_metric(metric, ylabel):

    plt.figure(figsize=(6,4))

    plt.plot(
        models["lambda"],
        models[metric],
        marker="o",
        linewidth=2,
        label="Neural controller"
    )

    plt.axhline(
        baseline[metric],
        linestyle="--",
        label="Expert PD"
    )

    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs $\\lambda$")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    output = f"oscillator/results/{metric}_vs_lambda.png"

    plt.savefig(output,dpi=300,bbox_inches="tight")

    print(f"Saved {output}")

    plt.close()




plot_metric("cum_reward","Cumulative Reward")

plot_metric("settling_time","Settling Time")

plot_metric("control_cost","Control Cost")

plot_metric("success","Success Rate")