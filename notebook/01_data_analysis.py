# Data Analysis Notebook.ipynb

# %% [markdown]
# # Recommendation System Results Analysis
#
# This notebook analyzes the results of our recommendation system experiments.

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# %% [markdown]
# ## Load Results Data
#
# First, let's load the evaluation results from the JSON file.

# %% [code]
# Load results
with open('results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Extract model names
models = list(results.keys())
print(f"Models: {models}")

# %% [markdown]
# ## Model Comparison
#
# Let's compare the performance of different models across key metrics.

# %% [code]
# Extract metrics
metrics = []
for model, model_results in results.items():
    for metric in model_results.keys():
        if isinstance(model_results[metric], (float, int)) and metric != 'training_time':
            metrics.append(metric)
metrics = sorted(list(set(metrics)))

# Create DataFrame for comparison
data = []
for model in models:
    for metric in metrics:
        if metric in results[model]:
            data.append({
                'Model': model,
                'Metric': metric,
                'Value': results[model][metric]
            })

df = pd.DataFrame(data)

# %% [code]
# Pivot the DataFrame for easier visualization
pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
pivot_df.fillna(0, inplace=True)

# %% [markdown]
# ### Precision and Recall Comparison

# %% [code]
# Filter for precision and recall metrics
precision_df = pivot_df[pivot_df.index.str.startswith('precision')]
recall_df = pivot_df[pivot_df.index.str.startswith('recall')]

# Plot precision comparison
plt.figure(figsize=(12, 6))
precision_df.plot(kind='bar', figsize=(12, 6))
plt.title('Precision Comparison')
plt.ylabel('Precision')
plt.xlabel('k')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot recall comparison
plt.figure(figsize=(12, 6))
recall_df.plot(kind='bar', figsize=(12, 6))
plt.title('Recall Comparison')
plt.ylabel('Recall')
plt.xlabel('k')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### NDCG and Hit Rate Comparison

# %% [code]
# Filter for NDCG and hit rate metrics
ndcg_df = pivot_df[pivot_df.index.str.startswith('ndcg')]
hit_rate_df = pivot_df[pivot_df.index.str.startswith('hit_rate')]

# Plot NDCG comparison
plt.figure(figsize=(12, 6))
ndcg_df.plot(kind='bar', figsize=(12, 6))
plt.title('NDCG Comparison')
plt.ylabel('NDCG')
plt.xlabel('k')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot hit rate comparison
plt.figure(figsize=(12, 6))
hit_rate_df.plot(kind='bar', figsize=(12, 6))
plt.title('Hit Rate Comparison')
plt.ylabel('Hit Rate')
plt.xlabel('k')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Segment Analysis
#
# Let's analyze the performance across different user segments.

# %% [code]
# Load segment results
segment_data = []
for model in models:
    if 'segments' in results[model]:
        for segment, segment_results in results[model]['segments'].items():
            for metric, value in segment_results.items():
                segment_data.append({
                    'Model': model,
                    'Segment': segment,
                    'Metric': metric,
                    'Value': value
                })

segment_df = pd.DataFrame(segment_data)

# %% [code]
# Compare segments for a specific metric (e.g., hit_rate@10)
metric = 'hit_rate@10'
segment_pivot = segment_df[segment_df['Metric'] == metric].pivot(
    index='Model', columns='Segment', values='Value'
)

plt.figure(figsize=(10, 6))
segment_pivot.plot(kind='bar', figsize=(10, 6))
plt.title(f'{metric} by Segment')
plt.ylabel('Value')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.legend(title='Segment')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cumulative Hit Rate Analysis

# %% [code]
# Extract k values from metrics
k_values = sorted(set([int(m.split('@')[1]) for m in metrics if '@' in m]))

# Plot cumulative hit rate
plt.figure(figsize=(10, 6))
for model in models:
    hit_rates = [(k, results[model].get(f'hit_rate@{k}', 0)) for k in k_values]
    hit_rates.sort()

    x_values = [x[0] for x in hit_rates]
    y_values = [x[1] for x in hit_rates]

    plt.plot(x_values, y_values, marker='o', label=model)

plt.xlabel('k')
plt.ylabel('Hit Rate')
plt.title('Cumulative Hit Rate by Model')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Training Time Comparison

# %% [code]
# Compare training times
training_times = []
for model in models:
    if 'training_time' in results[model]:
        training_times.append({
            'Model': model,
            'Training Time (s)': results[model]['training_time']
        })

training_df = pd.DataFrame(training_times)
if not training_df.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Training Time (s)', data=training_df)
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()