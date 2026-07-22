import pandas as pd
import ast

# Load CSV
df = pd.read_csv("oscillator/results/eval_results_2026_07_05_13_02_.csv")

# Convert string representations like "[0.0415]" to floats
df["mse"] = df["mse"].apply(lambda x: ast.literal_eval(x)[0])
df["settling_time"] = df["settling_time"].apply(lambda x: ast.literal_eval(x)[0])

# Create new column
df["non_mean_mse"] = df["mse"] * df["settling_time"]

# Save back to CSV
df.to_csv("your_file_with_non_mean_mse.csv", index=False)