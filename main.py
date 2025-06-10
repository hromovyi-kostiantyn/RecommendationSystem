import os
import sys
import argparse
import time
import psutil
import gc
from typing import List, Dict, Any

from config import load_config
from data.loaders import load_retail_dataset
from data.splitters import split_dataset
from models.popular import PopularityRecommender
from models.random_recommender import RandomRecommender
from models.collaborative.neighborhood import NeighborhoodRecommender
from models.collaborative.matrix_fact import MatrixFactorizationRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from evaluation.evaluator import Evaluator
from experiments.runner import ExperimentRunner, ExperimentAnalyzer
from utils.logger import get_logger
from utils.results import (
    create_experiment_directory,
    save_config,
    save_results_to_csv,
    save_results_to_json,
    generate_report
)
from utils.visualization import create_results_visualizations
from models.cold_start import ColdStartRecommender

logger = get_logger(__name__)

# Global variable to store training metrics
global_training_metrics = {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retail Recommendation System')

    # Configuration
    parser.add_argument('--config', type=str, default='config_base.yaml',
                        help='Configuration file to use (default: config_base.yaml)')

    # Mode of operation
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'recommend', 'experiment'],
                        default='train', help='Mode of operation')

    # Models to include
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['random', 'popularity', 'neighborhood', 'matrix_fact',
                                 'content_based', 'hybrid', 'all'],
                        default=['all'], help='Models to include')

    # Dataset options
    parser.add_argument('--data_dir', type=str, help='Directory containing dataset files')

    # Experiment options
    parser.add_argument('--experiment_id', type=str, help='ID of experiment to analyze')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')

    # Recommendation options
    parser.add_argument('--user_id', type=int, help='User ID for recommendations')
    parser.add_argument('--n_recommendations', type=int, default=10,
                        help='Number of recommendations to generate')

    return parser.parse_args()


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def get_model_size(model):
    """Estimate model size by measuring object size."""
    try:
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp_file:
            pickle.dump(model, tmp_file)
            size_bytes = tmp_file.tell()
        return size_bytes / 1024 / 1024  # Convert to MB
    except:
        return 0.0


def setup_models(model_types: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up the specified recommendation models.

    Args:
        model_types: List of model types to create
        config: Configuration dictionary

    Returns:
        Dictionary mapping model names to model instances
    """
    logger.info(f"Setting up models: {model_types}")

    # Expand 'all' to include all model types
    if 'all' in model_types:
        model_types = ['random', 'popularity', 'neighborhood', 'matrix_fact',
                       'content_based', 'hybrid']

    models = {}

    for model_type in model_types:
        logger.info(f"Creating {model_type} model")

        if model_type == 'random':
            models[model_type] = RandomRecommender(config)
        elif model_type == 'popularity':
            models[model_type] = PopularityRecommender(config)
        elif model_type == 'neighborhood':
            models[model_type] = NeighborhoodRecommender(config)
        elif model_type == 'matrix_fact':
            models[model_type] = MatrixFactorizationRecommender(config)
        elif model_type == 'content_based':
            models[model_type] = ContentBasedRecommender(config)
        elif model_type == 'hybrid':
            models[model_type] = HybridRecommender(config)

    return models


def train_models(models: Dict[str, Any], dataset):
    """
    Train the specified models on the dataset with resource monitoring.

    Args:
        models: Dictionary mapping model names to model instances
        dataset: Dataset to train on
    """
    logger.info("Training models with resource monitoring")

    # Get data for training
    train_data = dataset.get_interaction_data()
    user_features = dataset.get_user_features()
    item_features = dataset.get_item_features()
    user_segments = dataset.get_customer_segments()

    # Set ID mappings for all models
    for name, model in models.items():
        model.set_id_mappings(
            dataset.get_user_id_mapping(),
            dataset.get_item_id_mapping()
        )

    # Store training metrics for each model
    training_metrics = {}

    # Train each model
    for name, model in models.items():
        logger.info(f"Training {name} model")

        # Get initial memory state
        initial_memory = get_memory_usage()
        gc.collect()  # Clean up before training

        start_time = time.time()

        try:
            if name == 'content_based':
                model.fit(train_data, item_features, user_features)
            elif name == 'hybrid':
                # For hybrid model, first train component models
                component_models = {}

                # Create and train collaborative component
                if 'matrix_fact' in models:
                    component_models['collaborative'] = models['matrix_fact']
                else:
                    collaborative = MatrixFactorizationRecommender(config)
                    collaborative.set_id_mappings(
                        dataset.get_user_id_mapping(),
                        dataset.get_item_id_mapping()
                    )
                    collaborative.fit(train_data)
                    component_models['collaborative'] = collaborative

                # Create and train content-based component
                if 'content_based' in models:
                    component_models['content_based'] = models['content_based']
                else:
                    content_based = ContentBasedRecommender(config)
                    content_based.set_id_mappings(
                        dataset.get_user_id_mapping(),
                        dataset.get_item_id_mapping()
                    )
                    content_based.fit(train_data, item_features, user_features)
                    component_models['content_based'] = content_based

                # Add components to hybrid model
                for comp_name, comp_model in component_models.items():
                    model.add_recommender(comp_name, comp_model)

                # Complete setup of hybrid model
                model.fit(train_data)
            else:
                model.fit(train_data)

            end_time = time.time()
            training_time = end_time - start_time

            # Get final memory state
            final_memory = get_memory_usage()
            memory_delta = final_memory['rss_mb'] - initial_memory['rss_mb']

            # Get model size
            model_size = get_model_size(model)

            # Store metrics
            training_metrics[name] = {
                'training_time_seconds': training_time,
                'peak_memory_mb': final_memory['rss_mb'],
                'memory_delta_mb': memory_delta,
                'model_size_mb': model_size,
                'training_data_size': len(train_data),
                'num_users': len(dataset.get_user_id_mapping()),
                'num_items': len(dataset.get_item_id_mapping())
            }

            logger.info(f"Trained {name} model in {training_time:.2f} seconds")
            logger.info(f"Memory usage: {final_memory['rss_mb']:.2f} MB (delta: {memory_delta:+.2f} MB)")
            logger.info(f"Model size: {model_size:.2f} MB")

        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}", exc_info=True)
            training_metrics[name] = {
                'training_time_seconds': 0,
                'error': str(e)
            }

    # Store training metrics globally for use in evaluation
    global global_training_metrics
    global_training_metrics = training_metrics

    logger.info("Model training complete")
    return training_metrics


def evaluate_models(models: Dict[str, Any], dataset):
    """
    Evaluate the specified models on the dataset with resource metrics.

    Args:
        models: Dictionary mapping model names to model instances
        dataset: Dataset to evaluate on

    Returns:
        Dictionary with evaluation results including training metrics
    """
    logger.info("Evaluating models")

    # Split data for evaluation
    interaction_data = dataset.get_interaction_data()
    train_data, test_data = split_dataset(
        interaction_data,
        strategy=config.get('split.strategy', 'random'),
        test_size=config.get('split.test_size', 0.2),
        holdout_timestamp=config.get('split.holdout_timestamp'),
        random_state=config.get('data.random_seed', 42)
    )

    # Get additional data for evaluation
    item_features = dataset.get_item_features()
    user_features = dataset.get_user_features()
    user_segments = dataset.get_customer_segments()

    # Initialize evaluator
    evaluator = Evaluator(config)

    # Evaluate each model
    results = {}

    for name, model in models.items():
        if not model.is_fitted:
            logger.warning(f"Model {name} is not trained, skipping evaluation")
            continue

        logger.info(f"Evaluating {name} model")

        try:
            # Time the evaluation
            eval_start_time = time.time()

            model_results = evaluator.evaluate(
                model,
                test_data,
                item_features,
                user_features,
                user_segments
            )

            eval_end_time = time.time()
            evaluation_time = eval_end_time - eval_start_time

            # Add evaluation timing
            model_results['evaluation_time_seconds'] = evaluation_time

            # Test prediction speed
            sample_users = list(dataset.get_user_id_mapping().keys())[:5]
            sample_items = list(dataset.get_item_id_mapping().keys())[:5]

            if sample_users and sample_items:
                # Test individual predictions
                prediction_times = []
                successful_predictions = 0

                for _ in range(20):  # Test 20 predictions
                    try:
                        pred_start = time.time()
                        model.predict(sample_users[0], sample_items[0])
                        pred_end = time.time()
                        prediction_times.append((pred_end - pred_start) * 1000)  # Convert to ms
                        successful_predictions += 1
                    except:
                        pass

                if prediction_times:
                    model_results['avg_prediction_time_ms'] = sum(prediction_times) / len(prediction_times)
                    model_results['predictions_per_second'] = 1000 / model_results['avg_prediction_time_ms']

                # Test recommendation generation speed
                try:
                    rec_start = time.time()
                    model.recommend(sample_users[0], n=10)
                    rec_end = time.time()
                    model_results['recommendation_time_ms'] = (rec_end - rec_start) * 1000
                except:
                    model_results['recommendation_time_ms'] = 0

            # Add training metrics if available
            global global_training_metrics
            if 'global_training_metrics' in globals() and name in global_training_metrics:
                training_data = global_training_metrics[name]
                model_results.update(training_data)

            results[name] = model_results

            # Log key metrics
            logger.info(f"Results for {name} model:")
            logger.info(f"  Evaluation time: {evaluation_time:.2f}s")
            if 'training_time_seconds' in model_results:
                logger.info(f"  Training time: {model_results['training_time_seconds']:.2f}s")
            if 'avg_prediction_time_ms' in model_results:
                logger.info(f"  Avg prediction time: {model_results['avg_prediction_time_ms']:.2f}ms")
            for metric, value in model_results.items():
                if isinstance(value, (int, float)) and not metric.startswith('segment') and metric.endswith('@10'):
                    logger.info(f"  {metric}: {value:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating {name} model: {str(e)}", exc_info=True)

    experiment_dir = create_experiment_directory()

    save_config(config, experiment_dir)

    # Save results to CSV and JSON
    save_results_to_csv(results, experiment_dir)
    save_results_to_json(results, experiment_dir)

    # Create visualizations
    plots_dir = os.path.join(experiment_dir, 'plots')
    create_results_visualizations(results, plots_dir)

    # Generate report
    generate_report(results, experiment_dir)

    logger.info("Model evaluation complete")
    return results


def generate_recommendations(models: Dict[str, Any], dataset, user_id: int, n: int = 10):
    """
    Generate recommendations for a user from each model.

    Args:
        models: Dictionary mapping model names to model instances
        dataset: Dataset for context
        user_id: User ID to generate recommendations for
        n: Number of recommendations to generate

    Returns:
        Dictionary mapping model names to recommendation lists
    """
    logger.info(f"Generating recommendations for user {user_id}")

    # Get user's interactions for excluding seen items
    interaction_data = dataset.get_interaction_data()
    user_interactions = interaction_data[interaction_data['customer_key'] == user_id]
    seen_items = user_interactions['product_key'].tolist() if not user_interactions.empty else []

    # Get item features for displaying recommendations
    item_features = dataset.get_item_features()

    # Generate recommendations from each model
    all_recommendations = {}

    for name, model in models.items():
        if not model.is_fitted:
            logger.warning(f"Model {name} is not trained, skipping recommendations")
            continue

        logger.info(f"Generating recommendations from {name} model")

        try:
            recommendations = model.recommend(
                user_id,
                n=n,
                exclude_seen=True,
                seen_items=seen_items
            )

            all_recommendations[name] = recommendations

            # Log recommendations with item details if available
            logger.info(f"Top {len(recommendations)} recommendations from {name} model:")

            for i, (item_id, score) in enumerate(recommendations):
                item_name = item_features.loc[item_id]['product_name'] if item_id in item_features.index else 'Unknown'
                logger.info(f"  {i + 1}. {item_name} (ID: {item_id}, Score: {score:.4f})")

        except Exception as e:
            logger.error(f"Error generating recommendations from {name} model: {str(e)}", exc_info=True)

    logger.info("Recommendation generation complete")
    return all_recommendations


def run_experiment(config: Dict[str, Any], model_types: List[str], dataset=None):
    """
    Run a full experiment with multiple models.

    Args:
        config: Configuration dictionary
        model_types: List of model types to include
        dataset: Optional preloaded dataset

    Returns:
        Dictionary with experiment results
    """
    logger.info("Running experiment")

    # Initialize experiment runner
    runner = ExperimentRunner(config)

    # Load dataset if not provided
    if dataset is None:
        logger.info("Loading dataset...")
        dataset = load_retail_dataset(config)

        # Analyze and report matrix sparsity
        from utils.helpers import analyze_data_sparsity, print_sparsity_report

        # Get interaction data
        interaction_data = dataset.get_interaction_data()

        # Analyze sparsity
        sparsity_info = analyze_data_sparsity(interaction_data)

        # Log sparsity information
        logger.info(f"Matrix sparsity: {sparsity_info['sparsity_pct']:.2f}%")
        logger.info(f"Users with < 5 interactions: {sparsity_info['cold_start_users_pct']:.2f}%")
        logger.info(f"Items with < 5 interactions: {sparsity_info['cold_start_items_pct']:.2f}%")

        # Print detailed report
        if config.get('experiment.verbose', True):
            print_sparsity_report(sparsity_info)

        # Adjust models based on sparsity if enabled
        if config.get('experiment.auto_adjust', False) and sparsity_info['sparsity'] > 0.98:
            logger.info("Data is highly sparse. Adjusting model configurations...")

            # Increase regularization for matrix factorization
            if 'models.collaborative.matrix_fact.reg_all' in config:
                orig_reg = config.get('models.collaborative.matrix_fact.reg_all')
                new_reg = max(orig_reg, 0.1)  # Increase regularization for sparse data
                config['models.collaborative.matrix_fact.reg_all'] = new_reg
                logger.info(f"Adjusted matrix factorization regularization: {orig_reg} -> {new_reg}")

            # Ensure hybrid model includes cold-start handling
            if 'hybrid' in model_types and 'cold_start' not in model_types:
                model_types.append('cold_start')
                logger.info("Added cold-start model to complement hybrid model")

    # Run experiment with potentially adjusted configuration
    results = runner.run_experiment(model_types, dataset)

    logger.info(f"Experiment complete. Results saved to {runner.experiment_dir}")
    return results


def analyze_experiment(experiment_id: str, output_dir: str = None):
    """
    Analyze results from a previous experiment.

    Args:
        experiment_id: ID of the experiment to analyze
        output_dir: Optional directory to save analysis results
    """
    logger.info(f"Analyzing experiment {experiment_id}")

    # Determine experiment directory
    if output_dir:
        experiment_dir = os.path.join(output_dir, experiment_id)
    else:
        experiment_dir = os.path.join('results', experiment_id)

    # Check if experiment exists
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return

    # Initialize analyzer
    analyzer = ExperimentAnalyzer(experiment_dir)

    # Perform analysis
    logger.info("Comparing models")
    comparison = analyzer.compare_models()

    # Log comparison
    logger.info("Model comparison:")
    logger.info(f"\n{comparison}")

    # Create detailed plots
    logger.info("Creating detailed plots")
    plots_dir = os.path.join(experiment_dir, 'analysis')
    analyzer.create_detailed_plots(plots_dir)

    logger.info(f"Analysis complete. Results saved to {plots_dir}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config).config_data

    # Override config with command line arguments
    if args.data_dir:
        config['data']['dataset_dir'] = args.data_dir

    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir

    # Execute requested mode
    if args.mode == 'experiment':
        # Run full experiment
        results = run_experiment(config, args.models)

        if args.experiment_id:
            # Analyze experiment results
            analyze_experiment(args.experiment_id, args.output_dir)

    elif args.mode == 'train' or args.mode == 'evaluate' or args.mode == 'recommend':
        # Load dataset
        logger.info("Loading dataset")
        dataset = load_retail_dataset(config)

        # Set up models
        models = setup_models(args.models, config)

        # Train models if needed
        if args.mode == 'train' or not all(model.is_fitted for model in models.values()):
            train_models(models, dataset)

        # Evaluate models if requested
        if args.mode == 'evaluate':
            results = evaluate_models(models, dataset)

        # Generate recommendations if requested
        if args.mode == 'recommend':
            if args.user_id is None:
                logger.error("User ID required for recommendation mode")
                sys.exit(1)

            recommendations = generate_recommendations(
                models,
                dataset,
                args.user_id,
                args.n_recommendations
            )
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)