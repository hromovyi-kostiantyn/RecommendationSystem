# Time-Aware Recommendations Implementation Plan

## 1. Time Decay Functionality

### Implementation Steps:

1. **Create a time decay utility module**:

```python
# utils/time_utils.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional, List, Dict, Any

def calculate_time_decay(timestamp: Union[pd.Timestamp, datetime, str], 
                        half_life_days: float = 30.0,
                        reference_time: Optional[Union[pd.Timestamp, datetime, str]] = None) -> float:
    """
    Calculate time decay factor using exponential decay.
    
    Args:
        timestamp: Timestamp of the interaction
        half_life_days: Half-life in days (after this many days, the weight becomes 0.5)
        reference_time: Reference time for calculation (default: current time)
        
    Returns:
        Decay factor between 0 and 1
    """
    # Convert timestamp to datetime if needed
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
        
    # Set reference time to current time if not provided
    if reference_time is None:
        reference_time = datetime.now()
    elif isinstance(reference_time, str):
        reference_time = pd.to_datetime(reference_time)
        
    # Calculate time difference in days
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    if isinstance(reference_time, pd.Timestamp):
        reference_time = reference_time.to_pydatetime()
        
    time_diff_days = (reference_time - timestamp).total_seconds() / (24 * 3600)
    
    # Handle future timestamps
    if time_diff_days < 0:
        return 1.0
        
    # Calculate decay factor using exponential decay formula
    decay = np.exp(-time_diff_days * np.log(2) / half_life_days)
    
    return decay

def apply_time_decay_to_interactions(interactions_df: pd.DataFrame,
                                    timestamp_col: str = 'timestamp',
                                    rating_col: str = 'rating',
                                    half_life_days: float = 30.0,
                                    reference_time: Optional[Union[pd.Timestamp, datetime, str]] = None) -> pd.DataFrame:
    """
    Apply time decay to interaction ratings.
    
    Args:
        interactions_df: DataFrame with interactions
        timestamp_col: Name of timestamp column
        rating_col: Name of rating column
        half_life_days: Half-life in days
        reference_time: Reference time
        
    Returns:
        DataFrame with time-decayed ratings
    """
    # Make a copy to avoid modifying the original
    df = interactions_df.copy()
    
    # Ensure timestamp column is datetime
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Calculate decay factors
        df['time_decay'] = df[timestamp_col].apply(
            lambda ts: calculate_time_decay(ts, half_life_days, reference_time)
        )
        
        # Apply decay to ratings
        if rating_col in df.columns:
            df[f'original_{rating_col}'] = df[rating_col]
            df[rating_col] = df[rating_col] * df['time_decay']
    else:
        # If no timestamp column, add a default decay of 1.0
        df['time_decay'] = 1.0
        
    return df
```

2. **Create a time-aware recommender mixin**:

```python
# models/time_aware.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from utils.time_utils import apply_time_decay_to_interactions, calculate_time_decay

class TimeAwareMixin:
    """
    Mixin class to add time-awareness to recommender models.
    """
    
    def __init__(self, half_life_days: float = 30.0):
        """
        Initialize time-aware mixin.
        
        Args:
            half_life_days: Half-life in days for time decay
        """
        self.half_life_days = half_life_days
        self.reference_time = datetime.now()
        
    def apply_time_decay(self, interactions_df: pd.DataFrame,
                       timestamp_col: str = 'timestamp',
                       rating_col: str = 'rating') -> pd.DataFrame:
        """
        Apply time decay to interaction ratings.
        
        Args:
            interactions_df: DataFrame with interactions
            timestamp_col: Name of timestamp column
            rating_col: Name of rating column
            
        Returns:
            DataFrame with time-decayed ratings
        """
        return apply_time_decay_to_interactions(
            interactions_df=interactions_df,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            half_life_days=self.half_life_days,
            reference_time=self.reference_time
        )
        
    def update_reference_time(self, new_time: Optional[Union[datetime, str]] = None):
        """
        Update the reference time.
        
        Args:
            new_time: New reference time (default: current time)
        """
        if new_time is None:
            self.reference_time = datetime.now()
        elif isinstance(new_time, str):
            self.reference_time = pd.to_datetime(new_time)
        else:
            self.reference_time = new_time
```

3. **Apply the mixin to content-based recommender**:

```python
# models/content_based.py
from models.time_aware import TimeAwareMixin

class TimeAwareContentBasedRecommender(ContentBasedRecommender, TimeAwareMixin):
    """
    Time-aware content-based recommender that incorporates recency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time-aware content-based recommender.
        
        Args:
            config: Configuration dictionary
        """
        ContentBasedRecommender.__init__(self, config)
        
        # Get time-aware parameters from config
        time_config = config.get('models.time_aware', {})
        half_life_days = time_config.get('half_life_days', 30.0)
        
        TimeAwareMixin.__init__(self, half_life_days=half_life_days)
        
    def fit(self, train_data: pd.DataFrame, item_data: pd.DataFrame, user_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the time-aware content-based recommender.
        
        Args:
            train_data: DataFrame with training data
            item_data: DataFrame with item features
            user_data: Optional DataFrame with user features
        """
        # Apply time decay to interactions
        if 'timestamp' in train_data.columns:
            train_data_decayed = self.apply_time_decay(train_data)
        else:
            train_data_decayed = train_data
            
        # Call parent's fit method with decayed data
        super().fit(train_data_decayed, item_data, user_data)
        
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        # Call parent's predict method
        prediction = super().predict(user_id, item_id)
        
        # Apply recency boost for newer items if available
        if hasattr(self, 'item_features') and 'release_date' in self.item_features.columns:
            item = self.item_features.loc[item_id]
            if pd.notna(item['release_date']):
                release_date = pd.to_datetime(item['release_date'])
                # Calculate recency factor (newer items get a boost)
                recency_decay = calculate_time_decay(
                    release_date, 
                    half_life_days=180,  # Longer half-life for item novelty
                    reference_time=self.reference_time
                )
                recency_boost = 1.0 + (0.2 * recency_decay)  # Up to 20% boost for very new items
                prediction *= recency_boost
                
        return prediction
```

## 2. Session-Based Recommendations

### Implementation Steps:

1. **Create a session-based recommender class**:

```python
# models/sequential.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, Counter
from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)

class SessionBasedRecommender(BaseRecommender):
    """
    Session-based recommender that uses sequential patterns in user behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the session-based recommender.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("SessionBased", config)
        
        # Get config parameters
        session_config = config.get('models.session_based', {})
        self.session_length = session_config.get('session_length', 3)
        self.max_session_duration = session_config.get('max_session_duration', 30)  # minutes
        self.min_support = session_config.get('min_support', 2)
        
        # Transition matrices
        self.item_transitions = defaultdict(Counter)
        self.support_counts = Counter()
        
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the session-based recommender.
        
        Args:
            train_data: DataFrame with training data (must include timestamp)
        """
        logger.info("Fitting session-based recommender")
        
        # Require timestamp for sequential data
        if 'timestamp' not in train_data.columns and 'order_date' not in train_data.columns:
            raise ValueError("Session-based recommender requires timestamp column")
            
        # Use order_date if timestamp not available
        timestamp_col = 'timestamp' if 'timestamp' in train_data.columns else 'order_date'
        
        # Sort data by user and timestamp
        sorted_data = train_data.sort_values(['customer_key', timestamp_col])
        
        # Group by user
        user_groups = sorted_data.groupby('customer_key')
        
        # Process each user's sequence
        for user_id, group in user_groups:
            # Extract item sequence
            sequence = []
            last_timestamp = None
            
            for _, row in group.iterrows():
                current_item = row['product_key']
                current_time = row[timestamp_col]
                
                # If timestamp gap exceeds session duration, start a new session
                if last_timestamp is not None and pd.notna(current_time) and pd.notna(last_timestamp):
                    time_diff = (current_time - last_timestamp).total_seconds() / 60
                    if time_diff > self.max_session_duration:
                        # Process completed session
                        self._process_sequence(sequence)
                        sequence = []
                        
                # Add item to current session
                sequence.append(current_item)
                last_timestamp = current_time
                
                # If session reaches max length, process and slide window
                if len(sequence) >= self.session_length + 1:  # +1 for next item prediction
                    self._process_sequence(sequence)
                    sequence = sequence[-self.session_length:]  # Keep most recent items
                    
            # Process any remaining sequence
            if len(sequence) > 1:
                self._process_sequence(sequence)
                
        # Normalize transition probabilities
        self._normalize_transitions()
        
        self.is_fitted = True
        logger.info("Session-based recommender fitted successfully")
        
    def _process_sequence(self, sequence: List[int]) -> None:
        """
        Process a sequence to update transition counts.
        
        Args:
            sequence: List of sequential item IDs
        """
        # For each subsequence of length session_length, increment transition count
        for i in range(len(sequence) - 1):
            # Get context items (up to session_length)
            start_idx = max(0, i - self.session_length + 1)
            context = tuple(sequence[start_idx:i+1])
            next_item = sequence[i+1]
            
            # Update transition count
            self.item_transitions[context][next_item] += 1
            
            # Update support count for this context
            self.support_counts[context] += 1
            
    def _normalize_transitions(self) -> None:
        """
        Normalize transition counts to probabilities.
        """
        # Remove low-support contexts
        for context in list(self.item_transitions.keys()):
            if self.support_counts[context] < self.min_support:
                del self.item_transitions[context]
                
        # Convert counts to probabilities
        for context, next_items in self.item_transitions.items():
            total = sum(next_items.values())
            for next_item in next_items:
                next_items[next_item] /= total
                
    def predict(self, user_id: int, item_id: int, context: Optional[List[int]] = None) -> float:
        """
        Predict the likelihood of an item given context.
        
        Args:
            user_id: User ID (not used in this model)
            item_id: Target item ID
            context: Optional sequence of previous items
            
        Returns:
            Likelihood score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # If no context provided, try to find user's recent items
        if context is None:
            if hasattr(self, 'interactions'):
                user_interactions = self.interactions[self.interactions['customer_key'] == user_id]
                if len(user_interactions) > 0:
                    # Sort by timestamp if available
                    if 'timestamp' in user_interactions.columns:
                        user_interactions = user_interactions.sort_values('timestamp', ascending=False)
                    elif 'order_date' in user_interactions.columns:
                        user_interactions = user_interactions.sort_values('order_date', ascending=False)
                        
                    # Get most recent items
                    context = user_interactions['product_key'].head(self.session_length).tolist()
                    context.reverse()  # Reverse to get chronological order
                    
        # If still no context, return 0
        if not context:
            return 0.0
            
        # Convert context to tuple
        context_tuple = tuple(context[-self.session_length:])
        
        # If context not in transition matrix, try with shorter context
        while len(context_tuple) > 0:
            if context_tuple in self.item_transitions:
                # Return transition probability
                return self.item_transitions[context_tuple].get(item_id, 0.0)
                
            # Try with shorter context
            context_tuple = context_tuple[1:]
            
        # If no valid context found, return 0
        return 0.0
        
    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                 seen_items: Optional[List[int]] = None,
                 context: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate sequential recommendations based on context.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: List of seen items
            context: Optional sequence of previous items
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
            
        if n is None:
            n = self.top_k
            
        # If no context provided, try to find user's recent items
        if context is None:
            if hasattr(self, 'interactions'):
                user_interactions = self.interactions[self.interactions['customer_key'] == user_id]
                if len(user_interactions) > 0:
                    # Sort by timestamp if available
                    if 'timestamp' in user_interactions.columns:
                        user_interactions = user_interactions.sort_values('timestamp', ascending=False)
                    elif 'order_date' in user_interactions.columns:
                        user_interactions = user_interactions.sort_values('order_date', ascending=False)
                        
                    # Get most recent items
                    context = user_interactions['product_key'].head(self.session_length).tolist()
                    context.reverse()  # Reverse to get chronological order
                    
        # If still no context, fall back to popularity
        if not context:
            # Use global item popularity as fallback
            popularity = Counter()
            for transitions in self.item_transitions.values():
                for item, count in transitions.items():
                    popularity[item] += count
                    
            # Sort by popularity
            sorted_items = sorted(
                [(item, count) for item, count in popularity.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Filter out seen items if requested
            if exclude_seen and seen_items:
                sorted_items = [(item, score) for item, score in sorted_items if item not in seen_items]
                
            return sorted_items[:n]
            
        # Get recommendations based on context
        recommendations = []
        
        # Try with different context lengths
        context_tuple = tuple(context[-self.session_length:])
        while len(context_tuple) > 0:
            if context_tuple in self.item_transitions:
                # Get recommendations from this context
                for item, score in self.item_transitions[context_tuple].items():
                    if not exclude_seen or seen_items is None or item not in seen_items:
                        recommendations.append((item, score))
                        
                # If we have enough recommendations, break
                if len(recommendations) >= n:
                    break
                    
            # Try with shorter context
            context_tuple = context_tuple[1:]
            
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return recommendations[:n]
        
    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary with model-specific data.
        """
        return {
            'session_length': self.session_length,
            'max_session_duration': self.max_session_duration,
            'min_support': self.min_support,
            'item_transitions': dict(self.item_transitions),
            'support_counts': dict(self.support_counts)
        }
        
    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.
        
        Args:
            data: Dictionary with model-specific data.
        """
        self.session_length = data['session_length']
        self.max_session_duration = data['max_session_duration']
        self.min_support = data['min_support']
        
        # Convert dictionaries back to defaultdict(Counter)
        self.item_transitions = defaultdict(Counter)
        for context, next_items in data['item_transitions'].items():
            self.item_transitions[context].update(next_items)
            
        self.support_counts = Counter(data['support_counts'])
```

2. **Update config file to include time-aware settings**:

```yaml
# Add to your config_base.yaml file

# Time-aware recommendation settings
time_aware:
  half_life_days: 30.0  # Half-life for exponential decay
  use_time_decay: true  # Whether to apply time decay
  recency_boost: true   # Boost newer items

# Session-based recommendation settings
session_based:
  session_length: 3     # Number of items in a session
  max_session_duration: 30  # Maximum session duration in minutes
  min_support: 2        # Minimum support count for transitions
```

3. **Create a combined time-aware hybrid recommender**:

```python
# Create a new class in models/hybrid.py

class TimeAwareHybridRecommender(HybridRecommender, TimeAwareMixin):
    """
    Time-aware hybrid recommender that combines multiple recommendation approaches
    with temporal factors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time-aware hybrid recommender.
        
        Args:
            config: Configuration dictionary
        """
        HybridRecommender.__init__(self, config)
        
        # Get time-aware parameters from config
        time_config = config.get('models.time_aware', {})
        half_life_days = time_config.get('half_life_days', 30.0)
        
        TimeAwareMixin.__init__(self, half_life_days=half_life_days)
        
        # Flag to enable session-based component
        self.use_session_based = config.get('models.hybrid.use_session_based', False)
        
    def add_session_based_recommender(self, recommender):
        """
        Add a session-based recommender component.
        
        Args:
            recommender: SessionBasedRecommender instance
        """
        self.add_recommender('session_based', recommender, weight=0.2)
        self.use_session_based = True
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the time-aware hybrid recommender.
        
        Args:
            train_data: Training data
            **kwargs: Additional arguments
        """
        # Apply time decay to interactions
        if 'timestamp' in train_data.columns:
            train_data_decayed = self.apply_time_decay(train_data)
        else:
            train_data_decayed = train_data
            
        # Call parent's fit method with decayed data
        super().fit(train_data_decayed, **kwargs)
        
    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                 seen_items: Optional[List[int]] = None,
                 context: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate time-aware hybrid recommendations.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: List of seen items
            context: Optional sequence of previous items for session-based component
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
            
        if n is None:
            n = self.top_k
            
        # Get recommendations from all components
        all_recommendations = {}
        
        for name, recommender in self.recommenders.items():
            try:
                # Call session-based recommender with context if available
                if name == 'session_based' and self.use_session_based and context is not None:
                    recs = recommender.recommend(
                        user_id, n=n*2, exclude_seen=exclude_seen, 
                        seen_items=seen_items, context=context
                    )
                else:
                    recs = recommender.recommend(
                        user_id, n=n*2, exclude_seen=exclude_seen, seen_items=seen_items
                    )
                    
                all_recommendations[name] = recs
            except Exception as e:
                logger.warning(f"Error getting recommendations from {name}: {str(e)}")
                all_recommendations[name] = []
                
        # Combine recommendations using score fusion
        combined_scores = defaultdict(float)
        
        for name, recs in all_recommendations.items():
            weight = self.weights.get(name, 0.0)
            
            if weight > 0:
                # Normalize scores to 0-1 range
                scores = [score for _, score in recs]
                if scores:
                    max_score = max(scores)
                    min_score = min(scores)
                    score_range = max_score - min_score
                    
                    for item_id, score in recs:
                        if score_range > 0:
                            # Normalize and add weighted score
                            normalized_score = (score - min_score) / score_range
                            combined_scores[item_id] += weight * normalized_score
                        else:
                            # If all scores are the same, use 1.0
                            combined_scores[item_id] += weight
                            
        # Apply recency boost for newer items if enabled
        if hasattr(self, 'item_features') and 'release_date' in self.item_features.columns:
            for item_id in combined_scores:
                if item_id in self.item_features.index:
                    item = self.item_features.loc[item_id]
                    if pd.notna(item['release_date']):
                        release_date = pd.to_datetime(item['release_date'])
                        # Calculate recency factor (newer items get a boost)
                        recency_decay = calculate_time_decay(
                            release_date, 
                            half_life_days=180,  # Longer half-life for item novelty
                            reference_time=self.reference_time
                        )
                        recency_boost = 1.0 + (0.2 * recency_decay)  # Up to 20% boost for very new items
                        combined_scores[item_id] *= recency_boost
                        
        # Convert to list and sort by score
        recommendations = [(item_id, score) for item_id, score in combined_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return recommendations[:n]
```