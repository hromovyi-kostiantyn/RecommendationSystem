import os
import pandas as pd
from typing import Dict, Any, Optional, List, Union

from utils.logger import get_logger
from data.dataset import RetailDataset
from config import Config

logger = get_logger(__name__)


def load_retail_dataset(config: Union[Config, Dict[str, Any]]) -> RetailDataset:
    """
    Load the retail recommendation dataset with the specified configuration.

    This function creates a RetailDataset instance, loads the data, and prepares
    the interaction matrix for use in recommendation algorithms.

    Args:
        config: Configuration object or dictionary with dataset settings.

    Returns:
        Loaded RetailDataset instance.
    """
    logger.info("Loading retail dataset...")

    # Extract config data if a Config object was provided
    config_data = config.config_data if hasattr(config, 'config_data') else config

    # Create and load the dataset
    dataset = RetailDataset(config_data)
    dataset.load_data()

    # Create the interaction matrix
    dataset.create_interaction_matrix()

    # Log some statistics
    stats = dataset.get_statistics()
    logger.info(f"Loaded dataset with {stats.get('num_customers', 0)} customers, "
                f"{stats.get('num_products', 0)} products, and "
                f"{stats.get('num_interactions', 0)} interactions")

    if 'interaction_matrix_sparsity' in stats:
        logger.info(f"Interaction matrix sparsity: {stats['interaction_matrix_sparsity']:.4f}")

    return dataset


def get_dataset_split_options(config: Union[Config, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get dataset split options from config.

    Args:
        config: Configuration object or dictionary.

    Returns:
        Dictionary with split options.
    """
    # Extract config data if a Config object was provided
    if hasattr(config, 'config_data') and hasattr(config, 'get'):
        return {
            'test_size': config.get('split.test_size', 0.2),
            'strategy': config.get('split.strategy', 'random'),
            'random_seed': config.get('data.random_seed', 42),
            'holdout_timestamp': config.get('split.holdout_timestamp'),
            'min_ratings': config.get('split.min_ratings', 5)
        }

    # Handle dictionary config
    from data.dataset import get_nested_dict_value

    return {
        'test_size': get_nested_dict_value(config, 'split.test_size', 0.2),
        'strategy': get_nested_dict_value(config, 'split.strategy', 'random'),
        'random_seed': get_nested_dict_value(config, 'data.random_seed', 42),
        'holdout_timestamp': get_nested_dict_value(config, 'split.holdout_timestamp'),
        'min_ratings': get_nested_dict_value(config, 'split.min_ratings', 5)
    }


def load_data_with_config(config_name: str = "config_base.yaml") -> RetailDataset:
    """
    Load data with a specific configuration file.

    This function is a convenience wrapper that loads the configuration
    and then loads the dataset with that configuration.

    Args:
        config_name: Name of the configuration file to use.

    Returns:
        Loaded RetailDataset instance.
    """
    # Import here to avoid circular imports
    from config import load_config

    # Load configuration
    config = load_config(config_name)

    # Load the dataset
    return load_retail_dataset(config)