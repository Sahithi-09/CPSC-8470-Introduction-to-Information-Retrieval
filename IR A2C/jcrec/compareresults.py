import os
import json
import matplotlib.pyplot as plt
import re
import numpy as np

results_dir = "/home/jrajend/IR A2C/results"
models = ["greedy", "optimal", "a2c"]
results = {}

# Function to average metrics from multiple files
def average_metrics(files):
    metrics = {"new_attractiveness": [], "new_applicable_jobs": [], "avg_recommendation_time": []}
    for f in files:
        with open(os.path.join(results_dir, f), "r") as file:
            data = json.load(file)
            for key in metrics:
                metrics[key].append(data[key])
    return {key: float(np.mean(vals)) for key, vals in metrics.items()}

# Load model results
for model in models:
    files = [f for f in os.listdir(results_dir) if re.match(rf"final_{model}.*\.json", f)]
    if model == "a2c" and files:
        results[model] = average_metrics(files)
    elif files:
        with open(os.path.join(results_dir, files[0]), "r") as f:
            results[model] = json.load(f)
    else:
        print(f"⚠️  No result file found for model: {model}")

# Metrics and plotting
metrics = ["new_attractiveness", "new_applicable_jobs", "avg_recommendation_time"]
data = {metric: [results[m][metric] for m in models if m in results] for metric in metrics}

for metric in metrics:
    plt.figure()
    values = data[metric]
    models_present = [m for m in models if m in results]
    plt.bar(models_present, values)
    plt.title(f"{metric.replace('_', ' ').title()} Comparison")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel("Model")
    plt.grid(axis='y')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{metric}_comparison.png")
    plt.savefig(plot_path)
    print(f" Saved: {plot_path}")

# Print summary
print("\n📊 Summary:")
for metric in metrics:
    print(f"\n{metric.replace('_', ' ').title()}:")
    for i, m in enumerate([m for m in models if m in results]):
        print(f"  {m:8}: {data[metric][i]:.3f}")

print("\n Comparison completed.")
