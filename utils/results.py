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

    # Check if segment data exists
    has_segments = False
    for model_results in results.values():
        if 'segments' in model_results:
            has_segments = True
            break

    # Save segment data if it exists
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
        return {"metrics": metrics_path, "segments": segments_path}

    return {"metrics": metrics_path}


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
                            recall_metrics + ndcg_metrics + hit_rate_metrics])

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

    # Add training time comparison
    report.append("### Training Time\n")
    report.append("| Model | Training Time (seconds) |")
    report.append("|-------|--------------------------|")

    for model in models:
        training_time = results[model].get('training_time', "N/A")
        if isinstance(training_time, (int, float)):
            report.append(f"| {model} | {training_time:.2f} |")
        else:
            report.append(f"| {model} | {training_time} |")

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