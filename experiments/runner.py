import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from models.base import BaseRecommender
from models.popular import PopularityRecommender
from models.random_recommender import RandomRecommender
from models.collaborative.neighborhood import NeighborhoodRecommender
from models.collaborative.matrix_fact import MatrixFactorizationRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from evaluation.evaluator import Evaluator
from data.loaders import load_retail_dataset
from data.splitters import split_dataset
from utils.logger import get_logger
from utils.helpers import timer, save_object, load_object, ensure_directory, create_experiment_id

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Runner for recommendation system experiments.

    This class handles the execution of recommendation system experiments,
    including model training, evaluation, and results storage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment runner.

        Args:
            config: Dictionary with configuration parameters.
        """
        self.config = config
        self.output_dir = config.get('experiment.output_dir', 'results')
        self.save_models = config.get('experiment.save_models', True)
        self.verbose = config.get('experiment.verbose', True)
        self.experiment_id = create_experiment_id()

        # Create output directory
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_id)
        ensure_directory(self.experiment_dir)

        logger.info(f"Initialized ExperimentRunner with ID: {self.experiment_id}")

    def create_model(self, model_type: str) -> BaseRecommender:
        """
        Create a recommender model based on the specified type.

        Args:
            model_type: Type of recommender model to create.

        Returns:
            Initialized recommender model.
        """
        logger.info(f"Creating model of type: {model_type}")

        if model_type == 'random':
            return RandomRecommender(self.config)
        elif model_type == 'popularity':
            return PopularityRecommender(self.config)
        elif model_type == 'neighborhood':
            return NeighborhoodRecommender(self.config)
        elif model_type == 'matrix_fact':
            return MatrixFactorizationRecommender(self.config)
        elif model_type == 'content_based':
            return ContentBasedRecommender(self.config)
        elif model_type == 'hybrid':
            return HybridRecommender(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @timer
    def run_experiment(self, model_types: List[str], dataset=None) -> Dict[str, Any]:
        """
        Run an experiment with multiple recommendation models.

        Args:
            model_types: List of model types to evaluate.
            dataset: Optional preloaded dataset (will load if not provided).

        Returns:
            Dictionary with experiment results.
        """
        # Load dataset if not provided
        if dataset is None:
            logger.info("Loading dataset...")
            dataset = load_retail_dataset(self.config)

        # Split data
        logger.info("Splitting dataset...")
        interaction_data = dataset.get_interaction_data()
        train_data, test_data = split_dataset(
            interaction_data,
            strategy=self.config.get('split.strategy', 'random'),
            test_size=self.config.get('split.test_size', 0.2),
            holdout_timestamp=self.config.get('split.holdout_timestamp'),
            random_state=self.config.get('data.random_seed', 42)
        )

        # Initialize evaluator
        evaluator = Evaluator(self.config)

        # Track results
        all_results = {}
        trained_models = {}

        # Get user and item features
        user_features = dataset.get_user_features()
        item_features = dataset.get_item_features()

        # Get user segments
        user_segments = dataset.get_customer_segments()

        # Train and evaluate each model
        for model_type in model_types:
            logger.info(f"Running experiment for model: {model_type}")

            # Create model
            model = self.create_model(model_type)

            # Set ID mappings
            model.set_id_mappings(
                dataset.get_user_id_mapping(),
                dataset.get_item_id_mapping()
            )

            # Train model
            logger.info(f"Training {model_type} model...")
            start_time = time.time()

            if model_type == 'content_based':
                model.fit(train_data, item_features, user_features)
            elif model_type == 'hybrid':
                # For hybrid model, first create and train component models
                components = ['content_based', 'collaborative']
                component_models = {}

                for comp_type in components:
                    if comp_type == 'collaborative':
                        # Use matrix factorization for collaborative component
                        comp_model = self.create_model('matrix_fact')
                        comp_model.set_id_mappings(
                            dataset.get_user_id_mapping(),
                            dataset.get_item_id_mapping()
                        )
                        comp_model.fit(train_data)
                    elif comp_type == 'content_based':
                        comp_model = self.create_model('content_based')
                        comp_model.set_id_mappings(
                            dataset.get_user_id_mapping(),
                            dataset.get_item_id_mapping()
                        )
                        comp_model.fit(train_data, item_features, user_features)

                    component_models[comp_type] = comp_model

                # Add components to hybrid model
                for name, comp_model in component_models.items():
                    model.add_recommender(name, comp_model)

                # Complete setup of hybrid model
                model.fit(train_data)
            else:
                model.fit(train_data)

            training_time = time.time() - start_time

            # Evaluate model
            logger.info(f"Evaluating {model_type} model...")
            results = evaluator.evaluate(model, test_data, item_features, user_features, user_segments)

            # Add model type and training time to results
            results['model_type'] = model_type
            results['training_time'] = training_time

            # Save results
            all_results[model_type] = results
            trained_models[model_type] = model

            # Save model if requested
            if self.save_models:
                model_path = os.path.join(self.experiment_dir, f"{model_type}_model.pkl")
                model.save(model_path)
                logger.info(f"Saved {model_type} model to {model_path}")

        # Save overall results
        results_path = os.path.join(self.experiment_dir, "experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved experiment results to {results_path}")

        # Create summary plots
        self._create_results_plots(all_results)

        return all_results

    def _create_results_plots(self, results: Dict[str, Any]) -> None:
        """
        Create summary plots of experiment results.

        Args:
            results: Dictionary with experiment results.
        """
        logger.info("Creating results plots")

        plots_dir = os.path.join(self.experiment_dir, "plots")
        ensure_directory(plots_dir)

        # Extract metrics across models
        models = list(results.keys())
        metric_values = {}

        for model in models:
            for metric_name, value in results[model].items():
                if isinstance(value, (int, float)) and not metric_name.startswith('training_'):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append((model, value))

        # Create comparison plots for each metric
        for metric_name, values in metric_values.items():
            plt.figure(figsize=(10, 6))

            # Sort by performance
            values.sort(key=lambda x: x[1], reverse=True)

            model_names = [v[0] for v in values]
            metric_vals = [v[1] for v in values]

            # Create bar plot
            ax = sns.barplot(x=model_names, y=metric_vals)

            # Add labels
            for i, v in enumerate(metric_vals):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')

            # Set title and labels
            plt.title(f"Model Performance: {metric_name}")
            plt.ylabel(metric_name)
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(plots_dir, f"{metric_name}_comparison.png")
            plt.savefig(plot_path)
            plt.close()

        # Create training time comparison
        training_times = [(model, results[model].get('training_time', 0)) for model in models]
        training_times.sort(key=lambda x: x[1])

        plt.figure(figsize=(10, 6))
        model_names = [t[0] for t in training_times]
        times = [t[1] for t in training_times]

        ax = sns.barplot(x=model_names, y=times)

        # Add labels
        for i, v in enumerate(times):
            ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')

        # Set title and labels
        plt.title("Model Training Time Comparison")
        plt.ylabel("Training Time (seconds)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(plots_dir, "training_time_comparison.png")
        plt.savefig(plot_path)
        plt.close()

        # If there are user segments, create segment comparison plots
        for model in models:
            if 'segments' in results[model]:
                segments = results[model]['segments']

                # Find common metrics across segments
                common_metrics = set()
                for segment_name, segment_results in segments.items():
                    if not common_metrics:
                        common_metrics = set(segment_results.keys())
                    else:
                        common_metrics &= set(segment_results.keys())

                # Create plot for each metric
                for metric in common_metrics:
                    plt.figure(figsize=(10, 6))

                    segment_names = list(segments.keys())
                    segment_values = [segments[seg][metric] for seg in segment_names]

                    # Calculate overall value for comparison
                    overall_value = results[model].get(metric, None)

                    # Create bar plot
                    ax = sns.barplot(x=segment_names, y=segment_values)

                    # Add labels
                    for i, v in enumerate(segment_values):
                        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')

                    # Add overall line if available
                    if overall_value is not None:
                        plt.axhline(y=overall_value, color='r', linestyle='--',
                                    label=f"Overall: {overall_value:.3f}")
                        plt.legend()

                    # Set title and labels
                    plt.title(f"{model} - {metric} by User Segment")
                    plt.ylabel(metric)
                    plt.xlabel("Segment")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Save plot
                    plot_path = os.path.join(plots_dir, f"{model}_{metric}_segments.png")
                    plt.savefig(plot_path)
                    plt.close()

        logger.info(f"Saved {len(metric_values)} comparison plots to {plots_dir}")


class ExperimentAnalyzer:
    """
    Analyzer for recommendation system experiment results.

    This class provides methods to analyze and visualize results from
    recommendation system experiments, including model comparisons
    and performance breakdowns.
    """

    def __init__(self, experiment_dir: str = None):
        """
        Initialize the experiment analyzer.

        Args:
            experiment_dir: Path to experiment results directory.
        """
        self.experiment_dir = experiment_dir
        self.results = None

        if experiment_dir and os.path.exists(experiment_dir):
            results_path = os.path.join(experiment_dir, "experiment_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.results = json.load(f)

                logger.info(f"Loaded experiment results from {results_path}")

        logger.info("Initialized ExperimentAnalyzer")

    def load_results(self, results_path: str) -> None:
        """
        Load experiment results from a file.

        Args:
            results_path: Path to experiment results JSON file.
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        # Set experiment directory from results path
        self.experiment_dir = os.path.dirname(results_path)

        logger.info(f"Loaded experiment results from {results_path}")

    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare models based on specified metrics.

        Args:
            metrics: List of metrics to compare (if None, use all available metrics).

        Returns:
            DataFrame with model comparison.
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        # Get all models
        models = list(self.results.keys())

        # Get all metrics if not specified
        if metrics is None:
            metrics = set()
            for model in models:
                for metric in self.results[model].keys():
                    if isinstance(self.results[model][metric], (int, float)) and metric != 'training_time':
                        metrics.add(metric)
            metrics = sorted(list(metrics))

        # Create comparison DataFrame
        comparison = pd.DataFrame(index=models, columns=metrics)

        # Fill with results
        for model in models:
            for metric in metrics:
                if metric in self.results[model]:
                    comparison.loc[model, metric] = self.results[model][metric]

        # Add training time
        comparison['training_time'] = [self.results[model].get('training_time', None) for model in models]

        logger.info(f"Created model comparison with {len(models)} models and {len(metrics)} metrics")
        return comparison

    def get_best_model(self, metric: str) -> str:
        """
        Get the best performing model for a specific metric.

        Args:
            metric: Metric to use for comparison.

        Returns:
            Name of the best model.
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        # Extract metric values for each model
        model_metrics = {}
        for model in self.results:
            if metric in self.results[model]:
                model_metrics[model] = self.results[model][metric]

        if not model_metrics:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Find model with highest metric value
        best_model = max(model_metrics.items(), key=lambda x: x[1])[0]

        logger.info(f"Best model for {metric}: {best_model} with value {model_metrics[best_model]}")
        return best_model

    def segment_analysis(self, model: str, metrics: List[str] = None) -> pd.DataFrame:
        """
        Analyze performance across user segments for a specific model.

        Args:
            model: Model to analyze.
            metrics: List of metrics to compare (if None, use all available metrics).

        Returns:
            DataFrame with segment comparison.
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        if model not in self.results:
            raise ValueError(f"Model '{model}' not found in results")

        if 'segments' not in self.results[model]:
            raise ValueError(f"No segment results found for model '{model}'")

        segments = self.results[model]['segments']

        # Get all metrics if not specified
        if metrics is None:
            metrics = set()
            for segment in segments.values():
                for metric in segment.keys():
                    metrics.add(metric)
            metrics = sorted(list(metrics))

        # Create comparison DataFrame
        comparison = pd.DataFrame(index=list(segments.keys()), columns=metrics)

        # Fill with results
        for segment_name, segment_results in segments.items():
            for metric in metrics:
                if metric in segment_results:
                    comparison.loc[segment_name, metric] = segment_results[metric]

        # Add overall results for comparison
        overall = {}
        for metric in metrics:
            if metric in self.results[model]:
                overall[metric] = self.results[model][metric]

        comparison.loc['Overall'] = pd.Series(overall)

        logger.info(
            f"Created segment analysis for model '{model}' with {len(segments)} segments and {len(metrics)} metrics")
        return comparison

    def create_detailed_plots(self, output_dir: str = None) -> None:
        """
        Create detailed plots for further analysis.

        Args:
            output_dir: Directory to save plots (if None, use experiment_dir/detailed_plots).
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        if output_dir is None:
            output_dir = os.path.join(self.experiment_dir, "detailed_plots")

        ensure_directory(output_dir)

        # Create radar plots for model comparison
        self._create_radar_plots(output_dir)

        # Create segment performance heatmaps
        self._create_segment_heatmaps(output_dir)

        # Create metric correlation plot
        self._create_metric_correlation_plot(output_dir)

        logger.info(f"Created detailed plots in {output_dir}")

    def _create_radar_plots(self, output_dir: str) -> None:
        """
        Create radar plots for model comparison.

        Args:
            output_dir: Directory to save plots.
        """
        # Get models and metrics
        models = list(self.results.keys())

        # Get common metrics across all models
        common_metrics = set()
        for model in models:
            model_metrics = set()
            for metric, value in self.results[model].items():
                if isinstance(value, (int, float)) and metric != 'training_time' and not metric.startswith('model_'):
                    model_metrics.add(metric)

            if not common_metrics:
                common_metrics = model_metrics
            else:
                common_metrics &= model_metrics

        common_metrics = sorted(list(common_metrics))

        if not common_metrics:
            logger.warning("No common metrics found for radar plots")
            return

        # Create comparison DataFrame
        comparison = pd.DataFrame(index=models, columns=common_metrics)

        # Fill with results
        for model in models:
            for metric in common_metrics:
                comparison.loc[model, metric] = self.results[model][metric]

        # Normalize metrics to 0-1 range for radar plot
        normalized = pd.DataFrame(index=models, columns=common_metrics)

        for metric in common_metrics:
            min_val = comparison[metric].min()
            max_val = comparison[metric].max()

            if max_val > min_val:
                normalized[metric] = (comparison[metric] - min_val) / (max_val - min_val)
            else:
                normalized[metric] = 0.5  # Default if all values are the same

        # Create radar plot for each model
        for model in models:
            self._create_single_radar_plot(
                normalized.loc[model].values,
                common_metrics,
                model,
                os.path.join(output_dir, f"{model}_radar.png")
            )

        # Create combined radar plot
        self._create_combined_radar_plot(
            normalized,
            common_metrics,
            os.path.join(output_dir, "combined_radar.png")
        )

    def _create_single_radar_plot(self, values: List[float], categories: List[str], title: str,
                                  output_path: str) -> None:
        """
        Create a radar plot for a single model.

        Args:
            values: List of normalized metric values.
            categories: List of metric names.
            title: Plot title.
            output_path: Path to save the plot.
        """
        # Number of variables
        N = len(categories)

        # Repeat first value to close the polygon
        values = np.append(values, values[0])

        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.append(angles, angles[0])

        # Create figure
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Set y-axis limits
        ax.set_ylim(0, 1)

        # Add title
        plt.title(title, size=15, pad=20)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _create_combined_radar_plot(self, df: pd.DataFrame, categories: List[str], output_path: str) -> None:
        """
        Create a combined radar plot for multiple models.

        Args:
            df: DataFrame with normalized metric values for each model.
            categories: List of metric names.
            output_path: Path to save the plot.
        """
        # Number of variables
        N = len(categories)

        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

        # Create figure
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)

        # Plot data for each model
        for i, model in enumerate(df.index):
            values = df.loc[model].values
            values = np.append(values, values[0])
            angles_plot = np.append(angles, angles[0])

            ax.plot(angles_plot, values, 'o-', linewidth=2, label=model)
            ax.fill(angles_plot, values, alpha=0.1)

        # Set category labels
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)

        # Set y-axis limits
        ax.set_ylim(0, 1)

        # Add legend
        plt.legend(loc='upper right')

        # Add title
        plt.title("Model Comparison", size=15, pad=20)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _create_segment_heatmaps(self, output_dir: str) -> None:
        """
        Create heatmaps showing model performance across user segments.

        Args:
            output_dir: Directory to save plots.
        """
        # Get models with segment results
        segmented_models = []
        for model in self.results:
            if 'segments' in self.results[model]:
                segmented_models.append(model)

        if not segmented_models:
            logger.warning("No segment results found for heatmaps")
            return

        # Get all segments
        all_segments = set()
        for model in segmented_models:
            all_segments.update(self.results[model]['segments'].keys())
        all_segments = sorted(list(all_segments))

        # Get common metrics across all segments
        common_metrics = set()
        for model in segmented_models:
            for segment in self.results[model]['segments']:
                segment_metrics = set(self.results[model]['segments'][segment].keys())

                if not common_metrics:
                    common_metrics = segment_metrics
                else:
                    common_metrics &= segment_metrics

        common_metrics = sorted(list(common_metrics))

        if not common_metrics:
            logger.warning("No common metrics found across segments for heatmaps")
            return

        # Create heatmap for each metric
        for metric in common_metrics:
            # Create DataFrame for this metric
            heatmap_data = pd.DataFrame(index=segmented_models, columns=all_segments)

            # Fill with results
            for model in segmented_models:
                segments = self.results[model]['segments']
                for segment in all_segments:
                    if segment in segments and metric in segments[segment]:
                        heatmap_data.loc[model, segment] = segments[segment][metric]

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f')

            plt.title(f"{metric} Performance by Segment")
            plt.ylabel("Model")
            plt.xlabel("Segment")
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"{metric}_segment_heatmap.png")
            plt.savefig(plot_path)
            plt.close()

    def _create_metric_correlation_plot(self, output_dir: str) -> None:
        """
        Create a correlation plot between different metrics.

        Args:
            output_dir: Directory to save plots.
        """
        # Get all models
        models = list(self.results.keys())

        # Get common metrics across all models
        common_metrics = set()
        for model in models:
            model_metrics = set()
            for metric, value in self.results[model].items():
                if isinstance(value, (int, float)) and metric != 'training_time' and not metric.startswith('model_'):
                    model_metrics.add(metric)

            if not common_metrics:
                common_metrics = model_metrics
            else:
                common_metrics &= model_metrics

        common_metrics = sorted(list(common_metrics))

        if len(common_metrics) <= 1:
            logger.warning("Not enough common metrics for correlation plot")
            return

        # Create metric DataFrame
        metric_data = pd.DataFrame(index=models, columns=common_metrics)

        # Fill with results
        for model in models:
            for metric in common_metrics:
                metric_data.loc[model, metric] = self.results[model][metric]

        # Calculate correlation
        corr = metric_data.corr()

        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')

        plt.title("Metric Correlation")
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "metric_correlation.png")
        plt.savefig(plot_path)
        plt.close()