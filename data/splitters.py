import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
import random

from utils.logger import get_logger

logger = get_logger(__name__)


def random_split(
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interaction data randomly into train and test sets.

    Args:
        interactions: DataFrame with user-item interactions.
        test_size: Proportion of interactions to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info(f"Performing random split with test_size={test_size}")

    train_df, test_df = train_test_split(
        interactions,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")
    return train_df, test_df


def temporal_split(
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        timestamp_col: str = 'timestamp',
        holdout_timestamp: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interaction data based on time, with most recent interactions as test set.

    Args:
        interactions: DataFrame with user-item interactions.
        test_size: Proportion of interactions to use for testing.
        timestamp_col: Column name containing timestamp information.
        holdout_timestamp: Optional specific timestamp to use as cutoff.

    Returns:
        Tuple of (train_df, test_df).
    """
    if timestamp_col not in interactions.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found, falling back to random split")
        return random_split(interactions, test_size)

    logger.info(f"Performing temporal split with test_size={test_size}")

    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_dtype(interactions[timestamp_col]):
        try:
            interactions[timestamp_col] = pd.to_datetime(interactions[timestamp_col])
        except Exception as e:
            logger.error(f"Failed to convert timestamp column to datetime: {str(e)}")
            logger.warning("Falling back to random split")
            return random_split(interactions, test_size)

    # Sort by timestamp
    sorted_interactions = interactions.sort_values(timestamp_col)

    if holdout_timestamp:
        # Split based on specific timestamp
        try:
            holdout_time = pd.to_datetime(holdout_timestamp)
            train_df = sorted_interactions[sorted_interactions[timestamp_col] < holdout_time]
            test_df = sorted_interactions[sorted_interactions[timestamp_col] >= holdout_time]

            # Check if split is too imbalanced
            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning("Timestamp split resulted in empty train or test set")
                logger.warning("Falling back to proportional temporal split")
                holdout_timestamp = None
            else:
                actual_test_size = len(test_df) / len(interactions)
                logger.info(f"Split at timestamp {holdout_timestamp}, resulting in test_size={actual_test_size:.4f}")
                return train_df, test_df

        except Exception as e:
            logger.error(f"Failed to use holdout timestamp: {str(e)}")
            logger.warning("Falling back to proportional temporal split")
            holdout_timestamp = None

    if not holdout_timestamp:
        # Split based on proportion
        split_idx = int(len(sorted_interactions) * (1 - test_size))
        train_df = sorted_interactions.iloc[:split_idx]
        test_df = sorted_interactions.iloc[split_idx:]

        logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")

    return train_df, test_df


def user_split(
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        user_col: str = 'customer_key',
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by users, with some users entirely in test set.

    This is useful for evaluating how well the system works for cold-start users.

    Args:
        interactions: DataFrame with user-item interactions.
        test_size: Proportion of users to put in test set.
        user_col: Column name containing user IDs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    if user_col not in interactions.columns:
        logger.warning(f"User column '{user_col}' not found, falling back to random split")
        return random_split(interactions, test_size)

    logger.info(f"Performing user-based split with test_size={test_size}")

    # Get unique users
    unique_users = interactions[user_col].unique()

    # Split users into train and test
    random.seed(random_state)
    test_users = random.sample(list(unique_users), int(len(unique_users) * test_size))

    # Split interactions based on users
    test_df = interactions[interactions[user_col].isin(test_users)]
    train_df = interactions[~interactions[user_col].isin(test_users)]

    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")
    logger.info(f"Train set has {interactions[user_col].nunique() - len(test_users)} users, "
                f"test set has {len(test_users)} users")

    return train_df, test_df


def item_split(
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        item_col: str = 'product_key',
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by items, with some items entirely in test set.

    This is useful for evaluating how well the system works for cold-start items.

    Args:
        interactions: DataFrame with user-item interactions.
        test_size: Proportion of items to put in test set.
        item_col: Column name containing item IDs.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    if item_col not in interactions.columns:
        logger.warning(f"Item column '{item_col}' not found, falling back to random split")
        return random_split(interactions, test_size)

    logger.info(f"Performing item-based split with test_size={test_size}")

    # Get unique items
    unique_items = interactions[item_col].unique()

    # Split items into train and test
    random.seed(random_state)
    test_items = random.sample(list(unique_items), int(len(unique_items) * test_size))

    # Split interactions based on items
    test_df = interactions[interactions[item_col].isin(test_items)]
    train_df = interactions[~interactions[item_col].isin(test_items)]

    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")
    logger.info(f"Train set has {interactions[item_col].nunique() - len(test_items)} items, "
                f"test set has {len(test_items)} items")

    return train_df, test_df


def leave_one_out_split(
        interactions: pd.DataFrame,
        user_col: str = 'customer_key',
        item_col: str = 'product_key',
        timestamp_col: Optional[str] = 'timestamp',
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out split, where for each user, one interaction is in test set.

    If timestamp column is provided, the most recent interaction for each user is used.
    Otherwise, a random interaction is selected.

    Args:
        interactions: DataFrame with user-item interactions.
        user_col: Column name containing user IDs.
        item_col: Column name containing item IDs.
        timestamp_col: Column name containing timestamp information.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    if user_col not in interactions.columns or item_col not in interactions.columns:
        logger.warning(f"Required columns not found, falling back to random split")
        return random_split(interactions, 0.2)

    logger.info("Performing leave-one-out split")

    # Initialize test set
    test_indices = []

    # Process each user
    for user_id in interactions[user_col].unique():
        # Get user's interactions
        user_interactions = interactions[interactions[user_col] == user_id]

        if len(user_interactions) <= 1:
            # If user has only one interaction, keep it in train set
            continue

        if timestamp_col in interactions.columns:
            # Use most recent interaction
            try:
                if not pd.api.types.is_datetime64_dtype(user_interactions[timestamp_col]):
                    user_interactions[timestamp_col] = pd.to_datetime(user_interactions[timestamp_col])

                # Get index of most recent interaction
                most_recent_idx = user_interactions[timestamp_col].idxmax()
                test_indices.append(most_recent_idx)
            except Exception as e:
                logger.warning(f"Failed to use timestamp for user {user_id}: {str(e)}")
                # Fall back to random selection
                random_idx = user_interactions.sample(1, random_state=random_state).index[0]
                test_indices.append(random_idx)
        else:
            # Randomly select one interaction
            random_idx = user_interactions.sample(1, random_state=random_state).index[0]
            test_indices.append(random_idx)

    # Create train and test sets
    test_df = interactions.loc[test_indices]
    train_df = interactions.drop(test_indices)

    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")
    logger.info(f"Test set contains one interaction for {len(test_df)} users")

    return train_df, test_df


def stratified_split(
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        user_col: str = 'customer_key',
        item_col: str = 'product_key',
        strata_col: Optional[str] = None,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions with stratification to maintain the distribution of a column.

    Args:
        interactions: DataFrame with user-item interactions.
        test_size: Proportion of interactions to use for testing.
        user_col: Column name containing user IDs.
        item_col: Column name containing item IDs.
        strata_col: Column to use for stratification (e.g., 'category').
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    if strata_col is None or strata_col not in interactions.columns:
        logger.warning(f"Stratification column not specified or not found, falling back to random split")
        return random_split(interactions, test_size)

    logger.info(f"Performing stratified split on column '{strata_col}' with test_size={test_size}")

    # Perform stratified split
    train_df, test_df = train_test_split(
        interactions,
        test_size=test_size,
        stratify=interactions[strata_col],
        random_state=random_state
    )

    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test interactions")

    # Log stratification results
    train_strata = train_df[strata_col].value_counts(normalize=True)
    test_strata = test_df[strata_col].value_counts(normalize=True)

    for stratum in train_strata.index:
        if stratum in test_strata:
            logger.debug(f"Stratum '{stratum}': {train_strata[stratum]:.4f} in train, "
                         f"{test_strata[stratum]:.4f} in test")

    return train_df, test_df


def split_dataset(
        interactions: pd.DataFrame,
        strategy: str = 'random',
        test_size: float = 0.2,
        user_col: str = 'customer_key',
        item_col: str = 'product_key',
        timestamp_col: Optional[str] = 'timestamp',
        strata_col: Optional[str] = None,
        holdout_timestamp: Optional[str] = None,
        random_state: int = 42,
        min_ratings: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interaction data based on specified strategy.

    Args:
        interactions: DataFrame with user-item interactions.
        strategy: Splitting strategy ('random', 'time', 'user', 'item', 'leave_one_out', 'stratified').
        test_size: Proportion of interactions to use for testing.
        user_col: Column name containing user IDs.
        item_col: Column name containing item IDs.
        timestamp_col: Column name containing timestamp information.
        strata_col: Column to use for stratification.
        holdout_timestamp: Specific timestamp to use as cutoff for temporal split.
        random_state: Random seed for reproducibility.
        min_ratings: Minimum number of ratings per user for inclusion.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info(f"Splitting dataset with strategy: {strategy}")

    # Filter users with too few ratings if min_ratings is specified
    if min_ratings > 1:
        # Count ratings per user
        user_counts = interactions[user_col].value_counts()

        # Filter users with at least min_ratings
        valid_users = user_counts[user_counts >= min_ratings].index

        original_count = len(interactions)
        interactions = interactions[interactions[user_col].isin(valid_users)]

        if len(interactions) < original_count:
            logger.info(f"Filtered out {original_count - len(interactions)} interactions "
                        f"from users with fewer than {min_ratings} ratings")
            logger.info(f"Remaining interactions: {len(interactions)}")

    # Choose splitting strategy
    if strategy.lower() == 'random':
        return random_split(interactions, test_size, random_state)
    elif strategy.lower() == 'time':
        return temporal_split(interactions, test_size, timestamp_col, holdout_timestamp)
    elif strategy.lower() == 'user':
        return user_split(interactions, test_size, user_col, random_state)
    elif strategy.lower() == 'item':
        return item_split(interactions, test_size, item_col, random_state)
    elif strategy.lower() == 'leave_one_out':
        return leave_one_out_split(interactions, user_col, item_col, timestamp_col, random_state)
    elif strategy.lower() == 'stratified':
        return stratified_split(interactions, test_size, user_col, item_col, strata_col, random_state)
    else:
        logger.warning(f"Unknown splitting strategy: {strategy}, falling back to random split")
        return random_split(interactions, test_size, random_state)