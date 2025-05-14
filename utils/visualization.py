import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def create_results_visualizations(results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """
    Create and save visualizations of evaluation results.

    Args:
        results: Dictionary with evaluation results
        output_dir: Output directory for plots

    Returns:
        Dictionary mapping plot types to file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Track created plots
    plots = {}

    # Extract metrics and models
    metrics = set()
    models = list(results.keys())

    for model, model_results in results.items():
        for metric in model_results.keys():
            if isinstance(model_results[metric], (int, float)) and metric != 'training_time':
                metrics.add(metric)

    # Group metrics by type
    precision_metrics = sorted([m for m in metrics if m.startswith('precision')])
    recall_metrics = sorted([m for m in metrics if m.startswith('recall')])
    ndcg_metrics = sorted([m for m in metrics if m.startswith('ndcg')])
    hitrate_metrics = sorted([m for m in metrics if m.startswith('hit_rate')])
    other_metrics = sorted([m for m in metrics if m not in precision_metrics +
                            recall_metrics + ndcg_metrics + hitrate_metrics])

    # Create comparison charts for each metric group
    metric_groups = [
        ("Precision", precision_metrics),
        ("Recall", recall_metrics),
        ("NDCG", ndcg_metrics),
        ("Hit Rate", hitrate_metrics),
        ("Other Metrics", other_metrics)
    ]

    for title, metric_list in metric_groups:
        if not metric_list:
            continue

        plt.figure(figsize=(12, 8))

        # Create grouped bar chart
        bar_width = 0.8 / len(models)
        x = np.arange(len(metric_list))

        for i, model in enumerate(models):
            values = [results[model].get(metric, 0) for metric in metric_list]
            plt.bar(x + i * bar_width, values, width=bar_width, label=model)

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(f'{title} Comparison')
        plt.xticks(x + bar_width * (len(models) - 1) / 2,
                   [m.split('@')[0] + '@' + m.split('@')[1] if '@' in m else m for m in metric_list])
        plt.legend()
        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots[f"{title}_comparison"] = plot_path
        logger.info(f"Created {title} comparison plot: {plot_path}")

    # Create model comparison charts
    plt.figure(figsize=(14, 10))

    # Find common k values
    k_values = sorted(set([int(m.split('@')[1]) for m in metrics if '@' in m]))

    # For each k value, create a subplot comparing models
    num_rows = (len(k_values) + 1) // 2

    for i, k in enumerate(k_values):
        plt.subplot(num_rows, 2, i + 1)

        # Prepare data
        model_names = []
        precision_values = []
        recall_values = []

        for model in models:
            model_names.append(model)
            precision_values.append(results[model].get(f'precision@{k}', 0))
            recall_values.append(results[model].get(f'recall@{k}', 0))

        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, precision_values, width, label='Precision')
        plt.bar(x + width / 2, recall_values, width, label='Recall')

        plt.xlabel('Models')
        plt.ylabel('Value')
        plt.title(f'Precision and Recall @{k}')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plots["model_comparison"] = plot_path
    logger.info(f"Created model comparison plot: {plot_path}")

    # Create performance heatmap
    plt.figure(figsize=(12, 10))

    # Prepare data for heatmap
    heatmap_data = []
    for model in models:
        for metric in sorted(metrics):
            if metric.startswith(('precision', 'recall', 'ndcg', 'hit_rate', 'map', 'mrr')):
                heatmap_data.append({
                    "Model": model,
                    "Metric": metric,
                    "Value": results[model].get(metric, 0)
                })

    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data)
    pivot_df = df.pivot(index="Metric", columns="Model", values="Value")

    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Performance Metrics Heatmap")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "performance_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plots["performance_heatmap"] = plot_path
    logger.info(f"Created performance heatmap: {plot_path}")

    # Create segment performance heatmaps
    for model in models:
        if 'segments' in results[model]:
            segments = results[model]['segments']
            segment_names = list(segments.keys())

            # Find common metrics across segments
            segment_metrics = set()
            for segment in segments.values():
                for metric in segment.keys():
                    segment_metrics.add(metric)

            # Filter for key metrics
            key_metrics = [m for m in segment_metrics if m.startswith(('ndcg@', 'hit_rate@', 'recall@'))]

            if not key_metrics:
                continue

            # Convert to a DataFrame for easier plotting
            heatmap_data = []

            for metric in sorted(key_metrics):
                for segment in segment_names:
                    heatmap_data.append({
                        "Metric": metric,
                        "Segment": segment,
                        "Value": segments[segment].get(metric, 0)
                    })

            df = pd.DataFrame(heatmap_data)
            pivot_df = df.pivot(index="Metric", columns="Segment", values="Value")

            # Create heatmap
            plt.figure(figsize=(10, 12))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f')
            plt.title(f'{model} - Segment Performance')
            plt.tight_layout()

            # Save figure
            plot_path = os.path.join(output_dir, f"{model}_segment_heatmap.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            plots[f"{model}_segment_heatmap"] = plot_path
            logger.info(f"Created segment heatmap for {model}: {plot_path}")

    # Create cumulative hit rate chart
    plt.figure(figsize=(10, 6))

    for model in models:
        # Get hit rates at different k values
        hit_rates = [(int(k.split('@')[1]), results[model].get(k, 0))
                     for k in metrics if k.startswith('hit_rate@')]
        hit_rates.sort()

        if hit_rates:
            x_values = [x[0] for x in hit_rates]
            y_values = [x[1] for x in hit_rates]

            plt.plot(x_values, y_values, marker='o', label=model)

    plt.xlabel('k')
    plt.ylabel('Hit Rate')
    plt.title('Cumulative Hit Rate by Model')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "cumulative_hit_rate.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plots["cumulative_hit_rate"] = plot_path
    logger.info(f"Created cumulative hit rate plot: {plot_path}")

    # Create training time comparison chart
    training_times = [(model, results[model].get('training_time', 0)) for model in models]
    # Filter out models with missing training time
    training_times = [(model, time) for model, time in training_times if time > 0]

    if training_times:
        plt.figure(figsize=(10, 6))

        # Sort by training time
        training_times.sort(key=lambda x: x[1])

        models = [t[0] for t in training_times]
        times = [t[1] for t in training_times]

        plt.barh(models, times)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Model')
        plt.title('Training Time Comparison')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "training_time.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["training_time"] = plot_path
        logger.info(f"Created training time plot: {plot_path}")

    return plots