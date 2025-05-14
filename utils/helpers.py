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