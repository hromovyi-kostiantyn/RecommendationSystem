import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func: Function to measure.

    Returns:
        Wrapped function that logs execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {duration:.4f} seconds")
        return result

    return wrapper


def save_object(obj: Any, filepath: str) -> None:
    """
    Save an object to disk using pickle.

    Args:
        obj: Object to save.
        filepath: Path to save the object.
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

        logger.debug(f"Object saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {str(e)}")
        raise


def load_object(filepath: str) -> Any:
    """
    Load an object from disk using pickle.

    Args:
        filepath: Path to the saved object.

    Returns:
        Loaded object.
    """
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        logger.debug(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {str(e)}")
        raise


def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Created directory: {directory}")


def get_timestamp() -> str:
    """
    Get a formatted timestamp string.

    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_id(prefix: str = "experiment") -> str:
    """
    Create a unique experiment ID with timestamp.

    Args:
        prefix: Prefix for the ID.

    Returns:
        Unique experiment ID.
    """
    return f"{prefix}-{get_timestamp()}"


def check_memory_usage(df: pd.DataFrame) -> str:
    """
    Calculate and format the memory usage of a DataFrame.

    Args:
        df: DataFrame to check.

    Returns:
        Formatted memory usage string.
    """
    memory_usage = df.memory_usage(deep=True).sum()

    # Convert to appropriate unit
    if memory_usage < 1024:
        return f"{memory_usage} bytes"
    elif memory_usage < 1024 ** 2:
        return f"{memory_usage / 1024:.2f} KB"
    elif memory_usage < 1024 ** 3:
        return f"{memory_usage / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_usage / (1024 ** 3):.2f} GB"


def generate_user_item_interaction_matrix(
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        rating_col: Optional[str] = None,
        normalize: bool = False
) -> pd.DataFrame:
    """
    Generate a user-item interaction matrix from transaction data.

    Args:
        df: DataFrame containing transaction data.
        user_col: Column name for user IDs.
        item_col: Column name for item IDs.
        rating_col: Column name for ratings or interactions.
        normalize: Whether to normalize ratings by user.

    Returns:
        User-item interaction matrix as a DataFrame.
    """
    if rating_col:
        # Use provided rating values
        interaction_matrix = df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            aggfunc='mean',
            fill_value=0
        )
    else:
        # Create a matrix with just presence/absence (1/0)
        interaction_matrix = df.groupby([user_col, item_col]).size().unstack().fillna(0)
        interaction_matrix = interaction_matrix.astype(int)

    if normalize:
        # Normalize ratings by user
        user_means = interaction_matrix.mean(axis=1)
        interaction_matrix = interaction_matrix.sub(user_means, axis=0)

    logger.debug(f"Generated user-item matrix with shape {interaction_matrix.shape}")
    return interaction_matrix


def calculate_sparsity(matrix: pd.DataFrame) -> float:
    """
    Calculate the sparsity of a matrix.

    Args:
        matrix: Matrix to calculate sparsity for.

    Returns:
        Sparsity as a percentage.
    """
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero_elements = (matrix != 0).sum().sum()
    sparsity = 100 * (1 - non_zero_elements / total_elements)

    logger.debug(f"Matrix sparsity: {sparsity:.2f}%")
    return sparsity


def analyze_data_sparsity(interactions_df, user_col='customer_key', item_col='product_key'):
    """
    Analyze the sparsity of user-item interactions.

    Args:
        interactions_df: DataFrame with user-item interactions
        user_col: Name of user ID column
        item_col: Name of item ID column

    Returns:
        Dictionary with sparsity metrics
    """
    # Count unique users and items
    n_users = interactions_df[user_col].nunique()
    n_items = interactions_df[item_col].nunique()

    # Count interactions
    n_interactions = len(interactions_df)

    # Calculate density and sparsity
    possible_interactions = n_users * n_items
    density = n_interactions / possible_interactions
    sparsity = 1.0 - density

    # Analyze interactions per user
    interactions_per_user = interactions_df.groupby(user_col).size()

    # Calculate distribution statistics
    interaction_stats = {
        'min': interactions_per_user.min(),
        'max': interactions_per_user.max(),
        'mean': interactions_per_user.mean(),
        'median': interactions_per_user.median(),
        'p25': interactions_per_user.quantile(0.25),
        'p75': interactions_per_user.quantile(0.75),
    }

    # Count users with few interactions
    cold_start_users = (interactions_per_user < 5).sum()
    cold_start_pct = cold_start_users / n_users * 100

    # Count items with few interactions
    interactions_per_item = interactions_df.groupby(item_col).size()
    cold_start_items = (interactions_per_item < 5).sum()
    cold_start_items_pct = cold_start_items / n_items * 100

    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
        'possible_interactions': possible_interactions,
        'density': density,
        'sparsity': sparsity,
        'sparsity_pct': sparsity * 100,
        'interactions_per_user': interaction_stats,
        'cold_start_users': cold_start_users,
        'cold_start_users_pct': cold_start_pct,
        'cold_start_items': cold_start_items,
        'cold_start_items_pct': cold_start_items_pct
    }


def print_sparsity_report(sparsity_info):
    """
    Print a formatted report about interaction sparsity.

    Args:
        sparsity_info: Dictionary with sparsity metrics from analyze_data_sparsity
    """
    print("\n=== INTERACTION MATRIX SPARSITY ANALYSIS ===")
    print(f"Users: {sparsity_info['n_users']:,}")
    print(f"Items: {sparsity_info['n_items']:,}")
    print(f"Interactions: {sparsity_info['n_interactions']:,}")
    print(f"Possible interactions: {sparsity_info['possible_interactions']:,}")
    print(f"Matrix density: {sparsity_info['density']:.6f} ({sparsity_info['density'] * 100:.4f}%)")
    print(f"Matrix sparsity: {sparsity_info['sparsity']:.6f} ({sparsity_info['sparsity'] * 100:.4f}%)")

    print("\n--- User Interaction Distribution ---")
    stats = sparsity_info['interactions_per_user']
    print(f"Min: {stats['min']:.1f} interactions per user")
    print(f"25th percentile: {stats['p25']:.1f} interactions per user")
    print(f"Median: {stats['median']:.1f} interactions per user")
    print(f"Mean: {stats['mean']:.1f} interactions per user")
    print(f"75th percentile: {stats['p75']:.1f} interactions per user")
    print(f"Max: {stats['max']:.1f} interactions per user")

    print(
        f"\nCold-start users (< 5 interactions): {sparsity_info['cold_start_users']:,} ({sparsity_info['cold_start_users_pct']:.2f}%)")
    print(
        f"Cold-start items (< 5 interactions): {sparsity_info['cold_start_items']:,} ({sparsity_info['cold_start_items_pct']:.2f}%)")

    print("\nRECOMMENDATIONS:")
    if sparsity_info['sparsity'] > 0.995:
        print("- Your data is EXTREMELY sparse (>99.5%). Consider using content-based or hybrid approaches")
        print("- Matrix factorization will likely struggle with this level of sparsity")
        print("- Consider adding more user and item features to help with cold-start problems")
    elif sparsity_info['sparsity'] > 0.98:
        print("- Your data is VERY sparse (>98%). This is challenging but manageable with proper techniques")
        print("- Hybrid models combining collaborative and content-based approaches are recommended")
        print("- Consider using dimensionality reduction techniques and robust regularization")
    elif sparsity_info['sparsity'] > 0.95:
        print("- Your data sparsity is MODERATE (95-98%). Most recommender algorithms should work")
        print("- Matrix factorization with proper regularization should perform well")
        print("- Consider ensemble methods for best results")
    else:
        print("- Your data density is GOOD (<95% sparsity). Most algorithms should work well")
        print("- Pure collaborative filtering approaches should be effective")


def analyze_user_behavior(interactions_df, user_col='customer_key', item_col='product_key',
                          rating_col=None, time_col=None):
    """
    Analyze user behavior patterns in the interaction data.

    Args:
        interactions_df: DataFrame with user-item interactions
        user_col: Name of user ID column
        item_col: Name of item ID column
        rating_col: Name of rating column (optional)
        time_col: Name of timestamp column (optional)

    Returns:
        Dictionary with user behavior analysis
    """
    analysis = {}

    # Basic user statistics
    user_stats = interactions_df.groupby(user_col).agg({
        item_col: 'count'  # Number of interactions per user
    }).rename(columns={item_col: 'interaction_count'})

    if rating_col and rating_col in interactions_df.columns:
        user_ratings = interactions_df.groupby(user_col)[rating_col].agg(['mean', 'std', 'count'])
        user_stats = user_stats.join(user_ratings)

    # User segmentation based on activity
    user_stats['activity_level'] = pd.cut(
        user_stats['interaction_count'],
        bins=[0, 5, 20, 50, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    analysis['user_segments'] = user_stats['activity_level'].value_counts().to_dict()
    analysis['user_stats_summary'] = user_stats.describe()

    # Time-based analysis if timestamp available
    if time_col and time_col in interactions_df.columns:
        interactions_df[time_col] = pd.to_datetime(interactions_df[time_col])

        # Daily interaction patterns
        interactions_df['hour'] = interactions_df[time_col].dt.hour
        interactions_df['day_of_week'] = interactions_df[time_col].dt.day_name()

        analysis['hourly_patterns'] = interactions_df.groupby('hour').size().to_dict()
        analysis['daily_patterns'] = interactions_df.groupby('day_of_week').size().to_dict()

        # User session analysis
        user_sessions = interactions_df.groupby(user_col)[time_col].agg(['min', 'max', 'count'])
        user_sessions['session_duration'] = (user_sessions['max'] - user_sessions['min']).dt.total_seconds() / 3600
        analysis['session_stats'] = user_sessions.describe()

    return analysis


def analyze_item_popularity(interactions_df, item_col='product_key', rating_col=None):
    """
    Analyze item popularity patterns.

    Args:
        interactions_df: DataFrame with user-item interactions
        item_col: Name of item ID column
        rating_col: Name of rating column (optional)

    Returns:
        Dictionary with item popularity analysis
    """
    analysis = {}

    # Basic item statistics
    item_stats = interactions_df.groupby(item_col).size().reset_index(name='interaction_count')

    if rating_col and rating_col in interactions_df.columns:
        item_ratings = interactions_df.groupby(item_col)[rating_col].agg(['mean', 'std', 'count'])
        item_stats = item_stats.set_index(item_col).join(item_ratings).reset_index()

    # Popularity distribution
    analysis['popularity_distribution'] = item_stats['interaction_count'].describe().to_dict()

    # Long tail analysis
    total_interactions = item_stats['interaction_count'].sum()
    item_stats_sorted = item_stats.sort_values('interaction_count', ascending=False)
    item_stats_sorted['cumulative_interactions'] = item_stats_sorted['interaction_count'].cumsum()
    item_stats_sorted['cumulative_pct'] = item_stats_sorted['cumulative_interactions'] / total_interactions

    # Find where 80% of interactions come from (Pareto principle)
    top_items_80pct = item_stats_sorted[item_stats_sorted['cumulative_pct'] <= 0.8].shape[0]
    analysis['pareto_80_20'] = {
        'top_items_count': top_items_80pct,
        'top_items_pct': (top_items_80pct / len(item_stats)) * 100,
        'interactions_pct': 80
    }

    # Popularity categories
    item_stats['popularity_level'] = pd.cut(
        item_stats['interaction_count'],
        bins=[0, 5, 20, 100, float('inf')],
        labels=['Niche', 'Moderate', 'Popular', 'Blockbuster']
    )

    analysis['popularity_segments'] = item_stats['popularity_level'].value_counts().to_dict()

    return analysis


def create_performance_summary(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comprehensive performance summary table.

    Args:
        results: Dictionary with model evaluation results

    Returns:
        DataFrame with performance summary
    """
    summary_data = []

    for model_name, model_results in results.items():
        row = {
            'Model': model_name,
            'Training Time (s)': model_results.get('training_time_seconds', 'N/A'),
            'Peak Memory (MB)': model_results.get('peak_memory_mb', 'N/A'),
            'Model Size (MB)': model_results.get('model_size_mb', 'N/A'),
            'Prediction Time (ms)': model_results.get('avg_prediction_time_ms', 'N/A'),
            'Predictions/sec': model_results.get('predictions_per_second', 'N/A'),
            'Evaluation Time (s)': model_results.get('evaluation_time_seconds', 'N/A'),
        }

        # Add performance metrics
        performance_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10', 'map@10', 'mrr']
        for metric in performance_metrics:
            if metric in model_results:
                row[metric.upper()] = model_results[metric]
            else:
                row[metric.upper()] = 'N/A'

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def find_optimal_models(results: Dict[str, Dict[str, Any]],
                        performance_weight: float = 0.7,
                        efficiency_weight: float = 0.3) -> Dict[str, str]:
    """
    Find optimal models based on different criteria.

    Args:
        results: Dictionary with model evaluation results
        performance_weight: Weight for performance metrics (0-1)
        efficiency_weight: Weight for efficiency metrics (0-1)

    Returns:
        Dictionary with optimal model recommendations
    """
    recommendations = {}

    models = list(results.keys())

    # Find best performers for different criteria

    # Best overall performance (NDCG@10)
    ndcg_scores = {model: results[model].get('ndcg@10', 0) for model in models}
    if ndcg_scores:
        best_performance = max(ndcg_scores.items(), key=lambda x: x[1])
        recommendations['Best Performance'] = f"{best_performance[0]} (NDCG@10: {best_performance[1]:.4f})"

    # Fastest training
    training_times = {model: results[model].get('training_time_seconds', float('inf'))
                      for model in models if results[model].get('training_time_seconds', 0) > 0}
    if training_times:
        fastest_training = min(training_times.items(), key=lambda x: x[1])
        recommendations['Fastest Training'] = f"{fastest_training[0]} ({fastest_training[1]:.2f}s)"

    # Lowest memory usage
    memory_usage = {model: results[model].get('peak_memory_mb', float('inf'))
                    for model in models if results[model].get('peak_memory_mb', 0) > 0}
    if memory_usage:
        lowest_memory = min(memory_usage.items(), key=lambda x: x[1])
        recommendations['Lowest Memory'] = f"{lowest_memory[0]} ({lowest_memory[1]:.2f} MB)"

    # Fastest prediction
    prediction_times = {model: results[model].get('avg_prediction_time_ms', float('inf'))
                        for model in models if results[model].get('avg_prediction_time_ms', 0) > 0}
    if prediction_times:
        fastest_prediction = min(prediction_times.items(), key=lambda x: x[1])
        recommendations['Fastest Prediction'] = f"{fastest_prediction[0]} ({fastest_prediction[1]:.2f}ms)"

    # Smallest model
    model_sizes = {model: results[model].get('model_size_mb', float('inf'))
                   for model in models if results[model].get('model_size_mb', 0) > 0}
    if model_sizes:
        smallest_model = min(model_sizes.items(), key=lambda x: x[1])
        recommendations['Smallest Model'] = f"{smallest_model[0]} ({smallest_model[1]:.2f} MB)"

    # Combined efficiency score
    efficiency_scores = {}
    for model in models:
        # Normalize metrics (lower is better for resource metrics)
        training_norm = 1 / (1 + results[model].get('training_time_seconds', 1))
        memory_norm = 1 / (1 + results[model].get('peak_memory_mb', 100))
        prediction_norm = 1 / (1 + results[model].get('avg_prediction_time_ms', 1))
        performance_norm = results[model].get('ndcg@10', 0)

        efficiency_score = (efficiency_weight * (training_norm + memory_norm + prediction_norm) / 3 +
                            performance_weight * performance_norm)
        efficiency_scores[model] = efficiency_score

    if efficiency_scores:
        best_overall = max(efficiency_scores.items(), key=lambda x: x[1])
        recommendations['Best Overall'] = f"{best_overall[0]} (Score: {best_overall[1]:.4f})"

    return recommendations


def export_results_to_excel(results: Dict[str, Dict[str, Any]], output_path: str):
    """
    Export results to a comprehensive Excel file with multiple sheets.

    Args:
        results: Dictionary with model evaluation results
        output_path: Path to save Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Performance summary
            summary_df = create_performance_summary(results)
            summary_df.to_excel(writer, sheet_name='Performance Summary', index=False)

            # Detailed metrics
            detailed_data = []
            for model, model_results in results.items():
                for metric, value in model_results.items():
                    if isinstance(value, (int, float)) and metric != 'segments':
                        detailed_data.append({
                            'Model': model,
                            'Metric': metric,
                            'Value': value
                        })

            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)

            # Segment analysis (if available)
            segment_data = []
            for model, model_results in results.items():
                if 'segments' in model_results:
                    for segment, segment_results in model_results['segments'].items():
                        for metric, value in segment_results.items():
                            segment_data.append({
                                'Model': model,
                                'Segment': segment,
                                'Metric': metric,
                                'Value': value
                            })

            if segment_data:
                segment_df = pd.DataFrame(segment_data)
                segment_df.to_excel(writer, sheet_name='Segment Analysis', index=False)

            # Resource usage
            resource_data = []
            for model, model_results in results.items():
                resource_metrics = ['training_time_seconds', 'peak_memory_mb', 'model_size_mb',
                                    'avg_prediction_time_ms', 'predictions_per_second', 'evaluation_time_seconds']
                row = {'Model': model}
                for metric in resource_metrics:
                    row[metric] = model_results.get(metric, 'N/A')
                resource_data.append(row)

            resource_df = pd.DataFrame(resource_data)
            resource_df.to_excel(writer, sheet_name='Resource Usage', index=False)

        logger.info(f"Results exported to Excel: {output_path}")

    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        raise


def validate_dataset(df: pd.DataFrame, user_col: str = 'customer_key',
                     item_col: str = 'product_key', rating_col: str = None) -> Dict[str, Any]:
    """
    Validate dataset quality and identify potential issues.

    Args:
        df: DataFrame with interaction data
        user_col: Name of user ID column
        item_col: Name of item ID column
        rating_col: Name of rating column (optional)

    Returns:
        Dictionary with validation results
    """
    validation = {
        'errors': [],
        'warnings': [],
        'info': [],
        'stats': {}
    }

    # Check required columns
    required_cols = [user_col, item_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation['errors'].append(f"Missing required columns: {missing_cols}")
        return validation

    # Basic statistics
    validation['stats']['total_interactions'] = len(df)
    validation['stats']['unique_users'] = df[user_col].nunique()
    validation['stats']['unique_items'] = df[item_col].nunique()

    # Check for missing values
    missing_users = df[user_col].isnull().sum()
    missing_items = df[item_col].isnull().sum()

    if missing_users > 0:
        validation['errors'].append(f"Found {missing_users} missing user IDs")
    if missing_items > 0:
        validation['errors'].append(f"Found {missing_items} missing item IDs")

    # Check for duplicate interactions
    duplicates = df.duplicated(subset=[user_col, item_col]).sum()
    if duplicates > 0:
        validation['warnings'].append(f"Found {duplicates} duplicate user-item interactions")

    # Check rating column if provided
    if rating_col and rating_col in df.columns:
        missing_ratings = df[rating_col].isnull().sum()
        if missing_ratings > 0:
            validation['warnings'].append(f"Found {missing_ratings} missing ratings")

        rating_range = df[rating_col].max() - df[rating_col].min()
        validation['stats']['rating_range'] = rating_range
        validation['stats']['rating_mean'] = df[rating_col].mean()

    # Check data distribution
    user_interactions = df.groupby(user_col).size()
    item_interactions = df.groupby(item_col).size()

    # Users with very few interactions
    few_interaction_users = (user_interactions < 3).sum()
    if few_interaction_users > validation['stats']['unique_users'] * 0.5:
        validation['warnings'].append(f"Over 50% of users have fewer than 3 interactions")

    # Items with very few interactions
    few_interaction_items = (item_interactions < 3).sum()
    if few_interaction_items > validation['stats']['unique_items'] * 0.5:
        validation['warnings'].append(f"Over 50% of items have fewer than 3 interactions")

    # Check for power users/items
    power_users = (user_interactions > user_interactions.quantile(0.99)).sum()
    power_items = (item_interactions > item_interactions.quantile(0.99)).sum()

    if power_users > 0:
        validation['info'].append(f"Found {power_users} power users (top 1% by interactions)")
    if power_items > 0:
        validation['info'].append(f"Found {power_items} power items (top 1% by interactions)")

    # Overall quality assessment
    error_count = len(validation['errors'])
    warning_count = len(validation['warnings'])

    if error_count == 0 and warning_count == 0:
        validation['overall'] = "EXCELLENT - No issues detected"
    elif error_count == 0 and warning_count <= 2:
        validation['overall'] = "GOOD - Minor warnings only"
    elif error_count == 0:
        validation['overall'] = "FAIR - Multiple warnings detected"
    else:
        validation['overall'] = "POOR - Errors detected that need fixing"

    return validation