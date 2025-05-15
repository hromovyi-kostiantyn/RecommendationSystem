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


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Create a unique experiment ID with timestamp.

    Args:
        prefix: Prefix for the ID.

    Returns:
        Unique experiment ID.
    """
    return f"{prefix}_{get_timestamp()}"


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


# Add to utils/helpers.py

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