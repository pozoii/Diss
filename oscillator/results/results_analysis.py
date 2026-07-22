import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

#csv_file = sys.argv[1]
#csv_file = "oscillator/results/eval_results_2026_07_03_12_01_.csv"
csv_file = "oscillator/results/your_file_with_non_mean_mse.csv"
df = pd.read_csv(csv_file)

for col in ["mse", "settling_time", "control_cost", "success",'non_mean_mse']:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip("[]")
        .astype(float)
    )

df["model"] = (
    df["model"]
    .astype(str)
    .str.strip("[]")
    .str.replace("'", "", regex=False)
)

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


# ----------------------------
# Helper plotting function
# ----------------------------

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
    plt.savefig(f"oscillator/results/{metric}_vs_lambda.png", dpi=300)

"""plot_metric("mse", "Mean Squared Error")
plot_metric("settling_time", "Settling Time")
plot_metric("control_cost", "Control Cost")
plot_metric("success", "Success Rate")"""
plot_metric('non_mean_mse', "MSE * Timesteps")

plt.show()