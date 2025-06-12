import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

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
                            recall_metrics + ndcg_metrics + hitrate_metrics and
                            not m.endswith('_seconds') and not m.endswith('_mb') and
                            not m.endswith('_ms') and m != 'predictions_per_second'])

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

    # Create training time comparison chart
    training_times = [(model, results[model].get('training_time_seconds', 0)) for model in models]
    # Filter out models with missing training time
    training_times = [(model, time) for model, time in training_times if time > 0]

    if training_times:
        plt.figure(figsize=(10, 6))

        # Sort by training time
        training_times.sort(key=lambda x: x[1])

        models_list = [t[0] for t in training_times]
        times = [t[1] for t in training_times]

        plt.barh(models_list, times)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Model')
        plt.title('Training Time Comparison')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "training_time.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["training_time"] = plot_path
        logger.info(f"Created training time plot: {plot_path}")

    # Create memory usage comparison
    memory_data = [(model, results[model].get('peak_memory_mb', 0),
                    results[model].get('model_size_mb', 0)) for model in models]
    memory_data = [(m, peak, size) for m, peak, size in memory_data if peak > 0 or size > 0]

    if memory_data:
        plt.figure(figsize=(12, 6))

        model_names = [m[0] for m in memory_data]
        peak_memory = [m[1] for m in memory_data]
        model_sizes = [m[2] for m in memory_data]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, peak_memory, width, label='Peak Memory (MB)', alpha=0.8)
        plt.bar(x + width / 2, model_sizes, width, label='Model Size (MB)', alpha=0.8)

        plt.xlabel('Models')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "memory_usage.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["memory_usage"] = plot_path
        logger.info(f"Created memory usage plot: {plot_path}")

    # Create prediction speed comparison
    pred_speed_data = [(model, results[model].get('avg_prediction_time_ms', 0)) for model in models]
    pred_speed_data = [(m, speed) for m, speed in pred_speed_data if speed > 0]

    if pred_speed_data:
        plt.figure(figsize=(10, 6))

        model_names = [m[0] for m in pred_speed_data]
        pred_times = [m[1] for m in pred_speed_data]

        plt.bar(model_names, pred_times, color='coral')
        plt.xlabel('Models')
        plt.ylabel('Prediction Time (ms)')
        plt.title('Prediction Speed Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "prediction_speed.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["prediction_speed"] = plot_path
        logger.info(f"Created prediction speed plot: {plot_path}")

    # Create efficiency scatter plot (Performance vs Resources)
    efficiency_data = []
    for model in models:
        ndcg = results[model].get('ndcg@10', 0)
        training_time = results[model].get('training_time_seconds', 0)
        memory = results[model].get('peak_memory_mb', 0)

        if ndcg > 0 and training_time > 0:
            efficiency_data.append((model, ndcg, training_time, memory))

    if efficiency_data:
        plt.figure(figsize=(12, 8))

        models_eff = [d[0] for d in efficiency_data]
        ndcg_scores = [d[1] for d in efficiency_data]
        train_times = [d[2] for d in efficiency_data]
        memory_usage = [d[3] for d in efficiency_data]

        # Create scatter plot with bubble size representing memory usage
        plt.scatter(train_times, ndcg_scores, s=[m / 2 for m in memory_usage],
                    alpha=0.6, c=range(len(models_eff)), cmap='viridis')

        # Add labels
        for i, model in enumerate(models_eff):
            plt.annotate(model, (train_times[i], ndcg_scores[i]),
                         xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Training Time (seconds)')
        plt.ylabel('NDCG@10')
        plt.title('Model Efficiency: Performance vs Training Time\n(Bubble size = Memory Usage)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "efficiency_analysis.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["efficiency_analysis"] = plot_path
        logger.info(f"Created efficiency analysis plot: {plot_path}")

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

    # Create throughput comparison chart
    throughput_data = [(model, results[model].get('predictions_per_second', 0)) for model in models]
    throughput_data = [(model, tps) for model, tps in throughput_data if tps > 0]

    if throughput_data:
        plt.figure(figsize=(10, 6))

        # Sort by throughput
        throughput_data.sort(key=lambda x: x[1], reverse=True)

        models_tps = [t[0] for t in throughput_data]
        tps_values = [t[1] for t in throughput_data]

        plt.bar(models_tps, tps_values, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('Predictions per Second')
        plt.title('Prediction Throughput Comparison')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for i, v in enumerate(tps_values):
            plt.text(i, v + max(tps_values) * 0.01, f'{v:.1f}', ha='center', va='bottom')

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "prediction_throughput.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plots["prediction_throughput"] = plot_path
        logger.info(f"Created prediction throughput plot: {plot_path}")

    # Create resource summary dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Training time vs NDCG@10
    training_data = [(model, results[model].get('training_time_seconds', 0),
                      results[model].get('ndcg@10', 0)) for model in models]
    training_data = [(m, t, n) for m, t, n in training_data if t > 0 and n > 0]

    if training_data:
        models_t, times_t, ndcg_t = zip(*training_data)
        ax1.scatter(times_t, ndcg_t, s=100, alpha=0.7)
        for i, model in enumerate(models_t):
            ax1.annotate(model, (times_t[i], ndcg_t[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_ylabel('NDCG@10')
        ax1.set_title('Training Efficiency')
        ax1.grid(True, alpha=0.3)

    # Memory vs Hit Rate@10
    memory_data = [(model, results[model].get('peak_memory_mb', 0),
                    results[model].get('hit_rate@10', 0)) for model in models]
    memory_data = [(m, mem, hr) for m, mem, hr in memory_data if mem > 0 and hr > 0]

    if memory_data:
        models_m, memory_m, hitrate_m = zip(*memory_data)
        ax2.scatter(memory_m, hitrate_m, s=100, alpha=0.7, color='red')
        for i, model in enumerate(models_m):
            ax2.annotate(model, (memory_m[i], hitrate_m[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Peak Memory (MB)')
        ax2.set_ylabel('Hit Rate@10')
        ax2.set_title('Memory Efficiency')
        ax2.grid(True, alpha=0.3)

    # Model size vs NDCG@10
    size_data = [(model, results[model].get('model_size_mb', 0),
                  results[model].get('ndcg@10', 0)) for model in models]
    size_data = [(m, s, n) for m, s, n in size_data if s > 0 and n > 0]

    if size_data:
        models_s, sizes_s, ndcg_s = zip(*size_data)
        ax3.scatter(sizes_s, ndcg_s, s=100, alpha=0.7, color='green')
        for i, model in enumerate(models_s):
            ax3.annotate(model, (sizes_s[i], ndcg_s[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Model Size (MB)')
        ax3.set_ylabel('NDCG@10')
        ax3.set_title('Storage Efficiency')
        ax3.grid(True, alpha=0.3)

    # Prediction time vs Hit Rate@10
    pred_data = [(model, results[model].get('avg_prediction_time_ms', 0),
                  results[model].get('hit_rate@10', 0)) for model in models]
    pred_data = [(m, p, h) for m, p, h in pred_data if p > 0 and h > 0]

    if pred_data:
        models_p, pred_times_p, hitrate_p = zip(*pred_data)
        ax4.scatter(pred_times_p, hitrate_p, s=100, alpha=0.7, color='purple')
        for i, model in enumerate(models_p):
            ax4.annotate(model, (pred_times_p[i], hitrate_p[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)
        ax4.set_xlabel('Prediction Time (ms)')
        ax4.set_ylabel('Hit Rate@10')
        ax4.set_title('Inference Efficiency')
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Resource Efficiency Analysis Dashboard', fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "resource_dashboard.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plots["resource_dashboard"] = plot_path
    logger.info(f"Created resource dashboard: {plot_path}")

    return plots