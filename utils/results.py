import os
import json
import csv
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def create_experiment_directory(experiment_id: Optional[str] = None) -> str:
    """
    Create a directory for experiment results.

    Args:
        experiment_id: Optional experiment identifier

    Returns:
        Path to the created directory
    """
    from datetime import datetime

    # Create base results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Generate experiment ID if not provided
    if experiment_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"experiment-{timestamp}"

    # Create experiment directory
    experiment_dir = os.path.join('results', experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def save_config(config: Dict[str, Any], experiment_dir: str) -> str:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        experiment_dir: Experiment directory

    Returns:
        Path to the saved file
    """
    import yaml

    # Create file path
    config_path = os.path.join(experiment_dir, 'config.yaml')

    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved configuration to {config_path}")
    return config_path


def save_results_to_csv(results: Dict[str, Dict[str, Any]], experiment_dir: str) -> Dict[str, str]:
    """
    Save evaluation results to CSV files.

    Args:
        results: Dictionary with evaluation results
        experiment_dir: Experiment directory

    Returns:
        Dictionary mapping file types to saved file paths
    """
    # Define file paths
    metrics_path = os.path.join(experiment_dir, 'evaluation_results.csv')
    segments_path = os.path.join(experiment_dir, 'segment_results.csv')
    resource_path = os.path.join(experiment_dir, 'resource_metrics.csv')

    # Prepare metrics data
    metrics_data = []
    metrics_headers = ["model", "metric", "value"]

    for model, model_results in results.items():
        for metric, value in model_results.items():
            if isinstance(value, (int, float)) and metric != 'segments':
                metrics_data.append([model, metric, value])

    # Write metrics data to CSV
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics_headers)
        writer.writerows(metrics_data)

    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Prepare resource metrics data
    resource_data = []
    resource_headers = ["model", "training_time_seconds", "peak_memory_mb", "model_size_mb",
                        "avg_prediction_time_ms", "predictions_per_second", "evaluation_time_seconds",
                        "ndcg@10", "hit_rate@10", "precision@10", "recall@10"]

    for model, model_results in results.items():
        row = [model]
        for header in resource_headers[1:]:  # Skip 'model' column
            row.append(model_results.get(header, "N/A"))
        resource_data.append(row)

    # Write resource data to CSV
    with open(resource_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(resource_headers)
        writer.writerows(resource_data)

    logger.info(f"Saved resource metrics to {resource_path}")

    # Check if segment data exists
    has_segments = False
    for model_results in results.values():
        if 'segments' in model_results:
            has_segments = True
            break

    # Save segment data if it exists
    saved_files = {"metrics": metrics_path, "resources": resource_path}

    if has_segments:
        segment_data = []
        segment_headers = ["model", "segment", "metric", "value"]

        for model, model_results in results.items():
            if 'segments' in model_results:
                for segment, segment_results in model_results['segments'].items():
                    for metric, value in segment_results.items():
                        segment_data.append([model, segment, metric, value])

        with open(segments_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(segment_headers)
            writer.writerows(segment_data)

        logger.info(f"Saved segment results to {segments_path}")
        saved_files["segments"] = segments_path

    return saved_files


def save_results_to_json(results: Dict[str, Dict[str, Any]], experiment_dir: str) -> str:
    """
    Save evaluation results to a JSON file.

    Args:
        results: Dictionary with evaluation results
        experiment_dir: Experiment directory

    Returns:
        Path to the saved file
    """
    # Define file path
    json_path = os.path.join(experiment_dir, 'evaluation_results.json')

    # Write data to JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved evaluation results to {json_path}")
    return json_path


def generate_report(results: Dict[str, Dict[str, Any]], experiment_dir: str) -> str:
    """
    Generate a Markdown report of experiment results.

    Args:
        results: Dictionary with evaluation results
        experiment_dir: Experiment directory

    Returns:
        Path to the generated report
    """
    from datetime import datetime

    # Define file path
    report_path = os.path.join(experiment_dir, 'experiment_report.md')

    # Create report content
    report = ["# Recommendation System Experiment Report\n"]

    # Add experiment info
    report.append("## Experiment Information\n")
    report.append(f"- Date: {datetime.now().strftime('%Y-%m-%d')}")
    report.append(f"- Time: {datetime.now().strftime('%H:%M:%S')}")
    report.append(f"- Experiment directory: {experiment_dir}\n")

    # Get all models and metrics
    models = list(results.keys())
    metrics = set()

    for model_results in results.values():
        for metric in model_results.keys():
            if isinstance(model_results[metric], (int, float)) and metric != 'segments':
                metrics.add(metric)

    # Group metrics by type
    precision_metrics = sorted([m for m in metrics if m.startswith('precision')])
    recall_metrics = sorted([m for m in metrics if m.startswith('recall')])
    ndcg_metrics = sorted([m for m in metrics if m.startswith('ndcg')])
    hit_rate_metrics = sorted([m for m in metrics if m.startswith('hit_rate')])
    other_metrics = sorted([m for m in metrics if m not in precision_metrics +
                            recall_metrics + ndcg_metrics + hit_rate_metrics and
                            not m.endswith('_seconds') and not m.endswith('_mb') and
                            not m.endswith('_ms') and m != 'predictions_per_second'])

    # Add model comparison
    report.append("## Model Comparison\n")

    # Create comparison table for each metric group
    metric_groups = [
        ("Precision", precision_metrics),
        ("Recall", recall_metrics),
        ("NDCG", ndcg_metrics),
        ("Hit Rate", hit_rate_metrics),
        ("Other Metrics", other_metrics)
    ]

    for title, metric_list in metric_groups:
        if not metric_list:
            continue

        report.append(f"### {title}\n")

        # Create table header
        header = "| Model | " + " | ".join(metric_list) + " |"
        separator = "|-------|" + "|".join(["-----" for _ in metric_list]) + "|"
        report.append(header)
        report.append(separator)

        # Add rows for each model
        for model in models:
            row = f"| {model} |"
            for metric in metric_list:
                value = results[model].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    row += f" {value:.4f} |"
                else:
                    row += f" {value} |"
            report.append(row)

        report.append("")

    # Add resource usage section
    report.append("### Training Time\n")
    report.append("| Model | Training Time (seconds) | Peak Memory (MB) | Model Size (MB) | Prediction Time (ms) |")
    report.append("|-------|--------------------------|------------------|-----------------|---------------------|")

    for model in models:
        training_time = results[model].get('training_time_seconds', "N/A")
        peak_memory = results[model].get('peak_memory_mb', "N/A")
        model_size = results[model].get('model_size_mb', "N/A")
        pred_time = results[model].get('avg_prediction_time_ms', "N/A")

        if isinstance(training_time, (int, float)):
            training_time = f"{training_time:.2f}"
        if isinstance(peak_memory, (int, float)):
            peak_memory = f"{peak_memory:.2f}"
        if isinstance(model_size, (int, float)):
            model_size = f"{model_size:.2f}"
        if isinstance(pred_time, (int, float)):
            pred_time = f"{pred_time:.2f}"

        report.append(f"| {model} | {training_time} | {peak_memory} | {model_size} | {pred_time} |")

    report.append("")

    # Add segment analysis
    report.append("## Segment Analysis\n")

    for model in models:
        if 'segments' in results[model]:
            report.append(f"### {model} Model\n")

            segments = results[model]['segments']
            segment_names = list(segments.keys())

            # Find common metrics across segments
            segment_metrics = set()
            for segment_results in segments.values():
                for metric in segment_results.keys():
                    segment_metrics.add(metric)

            # Group segment metrics
            seg_precision = sorted([m for m in segment_metrics if m.startswith('precision')])
            seg_recall = sorted([m for m in segment_metrics if m.startswith('recall')])
            seg_ndcg = sorted([m for m in segment_metrics if m.startswith('ndcg')])
            seg_hit_rate = sorted([m for m in segment_metrics if m.startswith('hit_rate')])

            # Create tables for each metric type
            for title, metrics in [
                ("Precision", seg_precision),
                ("Recall", seg_recall),
                ("NDCG", seg_ndcg),
                ("Hit Rate", seg_hit_rate)
            ]:
                if not metrics:
                    continue

                report.append(f"#### {title} by Segment\n")

                # Create table header
                header = "| Segment | " + " | ".join(metrics) + " |"
                separator = "|---------|" + "|".join(["-----" for _ in metrics]) + "|"
                report.append(header)
                report.append(separator)

                # Add rows for each segment
                for segment in segment_names:
                    row = f"| {segment} |"
                    for metric in metrics:
                        value = segments[segment].get(metric, "N/A")
                        if isinstance(value, (int, float)):
                            row += f" {value:.4f} |"
                        else:
                            row += f" {value} |"
                    report.append(row)

                report.append("")

    # Add visualizations section
    report.append("## Visualizations\n")
    report.append("The following visualizations have been generated:\n")

    # Get all plot files
    plots_dir = os.path.join(experiment_dir, 'plots')
    if os.path.exists(plots_dir):
        plot_files = os.listdir(plots_dir)
        for plot_file in plot_files:
            report.append(f"- [{plot_file}](plots/{plot_file})")

    report.append("\n")

    # Add conclusions and insights
    report.append("## Key Insights\n")

    # Best model overall
    best_models = {}
    for metric in metrics:
        if metric.startswith(('precision', 'recall', 'ndcg', 'hit_rate')) and '@10' in metric:
            best_model = max(models, key=lambda m: results[m].get(metric, 0))
            best_value = results[best_model].get(metric, 0)
            best_models[metric] = (best_model, best_value)

    if best_models:
        report.append("### Best Performing Models\n")
        for metric, (model, value) in best_models.items():
            report.append(f"- **{metric}**: {model} ({value:.4f})")
        report.append("")

    # Resource efficiency insights
    report.append("### Resource Efficiency\n")

    # Find most efficient models
    fastest_training = min(models, key=lambda m: results[m].get('training_time_seconds', float('inf')))
    lowest_memory = min(models, key=lambda m: results[m].get('peak_memory_mb', float('inf')))
    smallest_model = min(models, key=lambda m: results[m].get('model_size_mb', float('inf')))
    fastest_prediction = min(models, key=lambda m: results[m].get('avg_prediction_time_ms', float('inf')))

    if results[fastest_training].get('training_time_seconds', 0) > 0:
        training_time = results[fastest_training]['training_time_seconds']
        report.append(f"- **Fastest Training**: {fastest_training} ({training_time:.2f}s)")

    if results[lowest_memory].get('peak_memory_mb', 0) > 0:
        memory_usage = results[lowest_memory]['peak_memory_mb']
        report.append(f"- **Lowest Memory Usage**: {lowest_memory} ({memory_usage:.2f} MB)")

    if results[smallest_model].get('model_size_mb', 0) > 0:
        model_size = results[smallest_model]['model_size_mb']
        report.append(f"- **Smallest Model**: {smallest_model} ({model_size:.2f} MB)")

    if results[fastest_prediction].get('avg_prediction_time_ms', 0) > 0:
        pred_time = results[fastest_prediction]['avg_prediction_time_ms']
        report.append(f"- **Fastest Prediction**: {fastest_prediction} ({pred_time:.2f} ms)")

    report.append("")

    # Segment differences
    report.append("### Segment Performance\n")
    report.append("Performance differences between user segments:\n")

    segment_metrics = ['hit_rate@10', 'recall@10', 'ndcg@10']

    for model in models:
        if 'segments' not in results[model]:
            continue

        segments = results[model]['segments']
        segment_names = list(segments.keys())

        # Skip if no common metrics
        common_metrics = set(segment_metrics) & set(segments[segment_names[0]].keys())
        if not common_metrics:
            continue

        report.append(f"**{model}**:")

        for metric in segment_metrics:
            if metric not in segments[segment_names[0]]:
                continue

            # Get values for each segment
            values = [(segment, segments[segment].get(metric, 0)) for segment in segment_names]
            values.sort(key=lambda x: x[1], reverse=True)

            # Format segment comparison
            segments_str = ", ".join([f"{segment}: {value:.4f}" for segment, value in values])
            report.append(f"- {metric}: {segments_str}")

        report.append("")

    # Write report to file
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    logger.info(f"Generated experiment report: {report_path}")
    return report_path