import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SUMMARY_DIR = "ball_on_plate/results/summaries"
FIGURE_DIR = "ball_on_plate/results/figures/summary"
PLOT_DIR = "ball_on_plate/results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DIFFICULTIES = [
    "easy",
    "medium",
    "hard",
    "very_hard",
    ]

DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "very_hard": "Very hard",
    }

METRICS = [
    ("success", "Success Rate"),
    ("failure", "Failure Rate"),
    ("mse", "Mean Squared Position Error"),
    ("settling_time", "Settling Time (steps)"),
    ("control_cost", "Control Effort"),
    ("final_error", "Final Position Error"),
    ]


# ============================================================
# Load all summaries
# ============================================================

files = [
    os.path.join(SUMMARY_DIR, f)
    for f in os.listdir(SUMMARY_DIR)
    if f.endswith(".csv")
]

dfs = [pd.read_csv(f) for f in files]

df = pd.concat(dfs, ignore_index=True)


# ============================================================
# Extract lambda
# ============================================================

def extract_lambda(model):

    if model == "Expert_MPC":
        return None

    match = re.search(r"lambda=(\d+(?:\.\d+)?)", model)

    if match is None:
        raise ValueError(f"Could not extract lambda from {model}")

    return float(match.group(1))


df["lambda"] = df["model"].apply(extract_lambda)

# ------------------------------------------------------------
# Helper: clean model names
# ------------------------------------------------------------

def model_label(row):

    if row["model"] == "Expert_MPC":
        return "Expert"

    return r"$\lambda$=" + str(row["lambda"])

df_plot = df.copy()
df_plot["model_label"] = df_plot.apply(model_label, axis=1)

# ============================================================
# Plot helper
# ============================================================

def plot_metric(metric, ylabel):

    plt.figure(figsize=(7, 5))

    for difficulty in DIFFICULTIES:

        subset = df[
            (df["difficulty"] == difficulty)
            &
            (df["model"] != "Expert_MPC")
        ].sort_values("lambda")

        plt.plot(
            subset["lambda"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=DIFFICULTY_LABELS[difficulty],
        )

        expert_value = df[
            (df["difficulty"] == difficulty)
            &
            (df["model"] == "Expert_MPC")
        ][metric].iloc[0]

        plt.axhline(
            expert_value,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)

    plt.title(f"{ylabel} vs $\\lambda$")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            f"{metric}_vs_lambda.png",
        ),
        dpi=300,
    )


# ============================================================
# Generate plots
# ============================================================

for metric, ylabel in METRICS:

    plot_metric(metric, ylabel)

plt.show()

# ============================================================
# Print comparison table
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for difficulty in DIFFICULTIES:

    print(f"\n{DIFFICULTY_LABELS[difficulty]}")

    table = (
        df[df["difficulty"] == difficulty]
        .sort_values("lambda", na_position="first")
        [[      
        "model",
        "success",
        "failure",
        "mse",
        "settling_time",
        "control_cost",
        "final_error",
        ]])

    print(table.to_string(index=False))

    subset = (df_plot[df_plot["difficulty"] == difficulty].sort_values("lambda", na_position="first"))

    plt.figure(figsize=(8,5))

    plt.bar(subset["model_label"],subset["success"])

    plt.ylim(0,1.05)

    plt.ylabel("Success Rate")

    plt.title(f"Success Rate - {DIFFICULTY_LABELS[difficulty]}")

    plt.xticks(rotation=45)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_DIR,f"success_{difficulty}.png"),dpi=300)

    subset = (df_plot[df_plot["difficulty"] == difficulty].sort_values("lambda", na_position="first"))

    plt.figure(figsize=(8,5))

    plt.bar(subset["model_label"],subset["control_cost"])

    plt.ylabel("Control Effort")

    plt.title(f"Control Effort - {DIFFICULTY_LABELS[difficulty]}")

    plt.xticks(rotation=45)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_DIR,f"control_effort_{difficulty}.png"),dpi=300)

# ------------------------------------------------------------
# 3. Success vs control effort Pareto plot
# ------------------------------------------------------------

plt.figure(figsize=(7,5))

for difficulty in DIFFICULTIES:

    subset = df_plot[
        df_plot["difficulty"] == difficulty
        ]

    plt.scatter(
        subset["control_cost"],
        subset["success"],
        s=80,
        label=DIFFICULTY_LABELS[difficulty]
        )

    for _, row in subset.iterrows():

        plt.annotate(row["model_label"],(row["control_cost"],row["success"]),fontsize=8)


plt.xlabel("Control Effort")

plt.ylabel("Success Rate")

plt.title("Success Rate vs Control Effort")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(FIGURE_DIR,"success_vs_control_effort.png"),dpi=300)

# ------------------------------------------------------------
# 4. Heatmap: success rate lambda vs difficulty
# ------------------------------------------------------------

success_table = (df_plot[df_plot["model"] != "Expert_MPC"].pivot(index="difficulty",columns="lambda",values="success"))

plt.figure(figsize=(7,4))

plt.imshow(success_table.values,aspect="auto")

plt.colorbar(label="Success Rate")

plt.xticks(range(len(success_table.columns)),success_table.columns)

plt.yticks(range(len(success_table.index)),[DIFFICULTY_LABELS[x] for x in success_table.index])

plt.xlabel(r"$\lambda$")

plt.ylabel("Difficulty")

plt.title("Success Rate Heatmap")

plt.tight_layout()

plt.savefig(os.path.join(FIGURE_DIR,"success_heatmap.png"),dpi=300)

# ------------------------------------------------------------
# 5. Heatmap: control effort lambda vs difficulty
# ------------------------------------------------------------

effort_table = (df_plot[df_plot["model"] != "Expert_MPC"].pivot(index="difficulty",columns="lambda",values="control_cost"))


plt.figure(figsize=(7,4))

plt.imshow(effort_table.values,aspect="auto")

plt.colorbar(label="Control Effort")

plt.xticks(range(len(effort_table.columns)), effort_table.columns)

plt.yticks(range(len(effort_table.index)),[DIFFICULTY_LABELS[x]for x in effort_table.index])

plt.xlabel(r"$\lambda$")

plt.ylabel("Difficulty")

plt.title("Control Effort Heatmap")

plt.tight_layout()

plt.savefig(os.path.join(FIGURE_DIR,"control_effort_heatmap.png"), dpi=300)
