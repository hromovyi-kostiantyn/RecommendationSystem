# Memory Efficiency Improvement Plan

## 1. Implement Sparse Matrix Operations

### Implementation Steps:

1. **Create a new utility file for sparse operations**:

```python
# utils/sparse_utils.py
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any

def convert_to_sparse_matrix(dense_matrix: np.ndarray) -> sp.csr_matrix:
    """Convert a dense matrix to sparse CSR format."""
    return sp.csr_matrix(dense_matrix)

def sparse_cosine_similarity(matrix_a: sp.csr_matrix, matrix_b: Optional[sp.csr_matrix] = None) -> np.ndarray:
    """
    Calculate cosine similarity between rows of sparse matrices.
    
    Args:
        matrix_a: First sparse matrix
        matrix_b: Second sparse matrix (if None, use matrix_a)
        
    Returns:
        Cosine similarity matrix
    """
    # If second matrix not provided, use the first one
    if matrix_b is None:
        matrix_b = matrix_a
        
    # Normalize the matrices
    matrix_a = normalize_sparse_matrix(matrix_a)
    matrix_b = normalize_sparse_matrix(matrix_b)
    
    # Calculate cosine similarity
    return (matrix_a * matrix_b.T).toarray()

def normalize_sparse_matrix(X: sp.csr_matrix) -> sp.csr_matrix:
    """
    Normalize a sparse matrix rows to unit length.
    
    Args:
        X: Sparse matrix to normalize
        
    Returns:
        Normalized sparse matrix
    """
    # Calculate L2 norm for each row
    norms = sp.linalg.norm(X, axis=1)
    
    # Avoid division by zero
    norms[norms == 0] = 1.0
    
    # Create a diagonal matrix with 1/norm values
    row_normalizer = sp.diags(1.0 / norms, 0)
    
    # Normalize the matrix
    return row_normalizer * X

def batch_similarity_calculation(user_vectors: sp.csr_matrix, 
                                 item_vectors: sp.csr_matrix,
                                 batch_size: int = 1000) -> np.ndarray:
    """
    Calculate similarity scores in batches to reduce memory usage.
    
    Args:
        user_vectors: Sparse user feature vectors
        item_vectors: Sparse item feature vectors
        batch_size: Number of users to process in each batch
        
    Returns:
        Similarity matrix (users x items)
    """
    num_users = user_vectors.shape[0]
    num_items = item_vectors.shape[0]
    
    # Initialize result matrix
    similarities = np.zeros((num_users, num_items))
    
    # Process in batches
    for i in range(0, num_users, batch_size):
        # Get batch of users
        end_idx = min(i + batch_size, num_users)
        user_batch = user_vectors[i:end_idx]
        
        # Calculate similarity for this batch
        batch_sim = (user_batch * item_vectors.T).toarray()
        
        # Store in result matrix
        similarities[i:end_idx] = batch_sim
        
    return similarities
```

2. **Modify the content-based recommender to use sparse matrices**:

```python
# models/content_based.py
# Add imports:
from utils.sparse_utils import convert_to_sparse_matrix, sparse_cosine_similarity, batch_similarity_calculation
import scipy.sparse as sp

# In the _create_item_feature_matrix method:
def _create_item_feature_matrix(self, item_features: pd.DataFrame) -> sp.csr_matrix:
    """
    Create a feature matrix for items based on text, categorical, and numerical features.
    
    Returns:
        Feature matrix as a sparse matrix.
    """
    # ... existing feature extraction code ...
    
    # Combine all feature matrices
    if feature_matrices:
        combined_matrix = sp.hstack(feature_matrices)
        logger.info(f"Created item feature matrix with shape {combined_matrix.shape}")
        return combined_matrix
    else:
        # If no features were processed successfully, create a dummy matrix
        logger.warning("No features were processed successfully, using dummy matrix")
        return sp.csr_matrix((len(item_features), 1))

# In the _calculate_item_similarity method:
def _calculate_item_similarity(self, feature_matrix: sp.csr_matrix) -> np.ndarray:
    """
    Calculate item similarity matrix based on feature matrix.
    """
    logger.info(f"Calculating item similarity using {self.similarity_metric} metric")

    if self.similarity_metric == 'cosine':
        # Calculate cosine similarity using sparse operations
        similarity = sparse_cosine_similarity(feature_matrix)
    else:
        # Default to cosine similarity
        logger.warning(f"Unknown similarity metric: {self.similarity_metric}, using cosine similarity")
        similarity = sparse_cosine_similarity(feature_matrix)

    # Set diagonal to 0 to avoid self-similarity
    np.fill_diagonal(similarity, 0)

    logger.info(f"Created item similarity matrix with shape {similarity.shape}")
    return similarity
```

3. **Update the user profile creation to be memory-efficient**:

```python
def _create_user_profiles(self, train_data: pd.DataFrame, user_col: str, item_col: str, rating_col: str) -> None:
    """
    Create user profiles based on their interactions with items.
    Uses batch processing to reduce memory usage.
    """
    logger.info("Creating user profiles")
    
    # Get unique user IDs and create empty profiles dictionary
    user_ids = train_data[user_col].unique()
    num_users = len(user_ids)
    num_items = self.item_similarity_matrix.shape[1]
    
    # Process users in batches to reduce memory consumption
    batch_size = 1000  # Adjust based on available memory
    
    # Initialize sparse user profiles matrix
    user_profiles = sp.lil_matrix((num_users, num_items))
    
    # Create a mapping from user IDs to indices
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    
    # Process in batches
    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        batch_user_ids = user_ids[start_idx:end_idx]
        
        for user_id in batch_user_ids:
            # Get user's index in the matrix
            user_idx = user_id_to_idx[user_id]
            
            # Get items rated by the user
            user_data = train_data[train_data[user_col] == user_id]
            
            if len(user_data) == 0:
                continue
                
            # Get item indices and ratings
            item_indices = []
            ratings = []
            
            for _, row in user_data.iterrows():
                item_id = row[item_col]
                
                # Check if item exists in our feature matrix
                if item_id in self.item_features.index:
                    try:
                        item_idx = self.item_features.index.get_loc(item_id)
                        item_indices.append(item_idx)
                        ratings.append(row[rating_col])
                    except:
                        continue
                        
            if not item_indices:
                continue
                
            # Convert to numpy arrays
            item_indices = np.array(item_indices)
            ratings = np.array(ratings)
            
            # Normalize ratings to sum to 1
            if np.sum(ratings) > 0:
                ratings = ratings / np.sum(ratings)
                
            # Create user profile as weighted average of item features
            for i, item_idx in enumerate(item_indices):
                user_profiles[user_idx] += ratings[i] * self.item_similarity_matrix[item_idx]
                
        logger.debug(f"Processed user profiles {start_idx+1} to {end_idx} of {num_users}")
        
    # Convert to CSR for efficient storage and operations
    self.user_profiles = user_profiles.tocsr()
    
    logger.info(f"Created {num_users} user profiles")
```

## 2. Implement Incremental Learning

### Implementation Steps:

1. **Create a base incremental learning class**:

```python
# models/incremental.py
from models.base import BaseRecommender
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
import time
from utils.logger import get_logger

logger = get_logger(__name__)

class IncrementalLearningMixin:
    """
    Mixin class that adds incremental learning capabilities to recommenders.
    """
    
    def partial_fit(self, new_data: pd.DataFrame) -> None:
        """
        Update the model with new data without retraining from scratch.
        
        Args:
            new_data: DataFrame with new interaction data
        """
        raise NotImplementedError("Subclasses must implement partial_fit")
        
    def fit_or_update(self, train_data: pd.DataFrame, 
                      force_retrain: bool = False, 
                      last_updated: Optional[float] = None,
                      update_threshold: int = 100) -> None:
        """
        Either fit the model from scratch or update incrementally.
        
        Args:
            train_data: Training data
            force_retrain: Whether to force a full retrain
            last_updated: Timestamp of last update
            update_threshold: Minimum number of new interactions to trigger update
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted or force_retrain:
            # Fit from scratch if not fitted or forced
            self.fit(train_data)
            self._set_last_updated()
        else:
            # Check if we need to update
            if last_updated is None and hasattr(self, 'last_updated'):
                last_updated = self.last_updated
                
            if last_updated is not None:
                # Get only new data since last update
                if 'timestamp' in train_data.columns:
                    new_data = train_data[train_data['timestamp'] > last_updated]
                    
                    if len(new_data) >= update_threshold:
                        self.partial_fit(new_data)
                        self._set_last_updated()
                        logger.info(f"Incrementally updated model with {len(new_data)} new interactions")
                    else:
                        logger.info(f"Skipping update, only {len(new_data)} new interactions")
                else:
                    logger.warning("Cannot perform incremental update without timestamp column")
            else:
                logger.warning("No last_updated timestamp available, performing full fit")
                self.fit(train_data)
                self._set_last_updated()
                
    def _set_last_updated(self) -> None:
        """Set the last updated timestamp to current time."""
        self.last_updated = time.time()
```

2. **Add incremental learning to content-based recommender**:

```python
# In models/content_based.py
from models.incremental import IncrementalLearningMixin

class ContentBasedRecommender(BaseRecommender, IncrementalLearningMixin):
    # ... existing code ...
    
    def partial_fit(self, new_data: pd.DataFrame) -> None:
        """
        Update the model with new interactions.
        
        Args:
            new_data: DataFrame with new interactions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before partial_fit")
            
        # Extract column names
        user_col = 'customer_key'
        item_col = 'product_key'
        rating_col = 'rating'
        
        if rating_col not in new_data.columns:
            new_data[rating_col] = 1.0
            
        # Create or update user profiles for users with new interactions
        for user_id in new_data[user_col].unique():
            # Get user's new interactions
            user_data = new_data[new_data[user_col] == user_id]
            
            # Extract item indices and ratings
            item_indices = []
            ratings = []
            
            for _, row in user_data.iterrows():
                item_id = row[item_col]
                
                if item_id in self.item_features.index:
                    try:
                        item_idx = self.item_features.index.get_loc(item_id)
                        item_indices.append(item_idx)
                        ratings.append(row[rating_col])
                    except:
                        continue
                        
            if not item_indices:
                continue
                
            # Convert to numpy arrays
            item_indices = np.array(item_indices)
            ratings = np.array(ratings)
            
            # Normalize ratings
            if np.sum(ratings) > 0:
                ratings = ratings / np.sum(ratings)
                
            # Update user profile
            if user_id in self.user_profiles:
                # Update existing profile (weighted update)
                old_profile = self.user_profiles[user_id]
                new_profile = np.zeros_like(old_profile)
                
                for i, item_idx in enumerate(item_indices):
                    new_profile += ratings[i] * self.item_similarity_matrix[item_idx]
                    
                # Combine old and new profile (70% old, 30% new)
                self.user_profiles[user_id] = 0.7 * old_profile + 0.3 * new_profile
            else:
                # Create new profile
                profile = np.zeros(self.item_similarity_matrix.shape[1])
                
                for i, item_idx in enumerate(item_indices):
                    profile += ratings[i] * self.item_similarity_matrix[item_idx]
                    
                self.user_profiles[user_id] = profile
                
        logger.info(f"Updated {len(new_data[user_col].unique())} user profiles")
```

3. **Implement memory-efficient batch prediction**:

```python
# Add to models/base.py
def recommend_batch(self, user_ids: List[int], n: int = None, 
                   exclude_seen: bool = True,
                   user_interactions: Optional[Dict[int, List[int]]] = None) -> Dict[int, List[Tuple[int, float]]]:
    """
    Generate recommendations for multiple users efficiently.
    
    Args:
        user_ids: List of user IDs
        n: Number of recommendations per user
        exclude_seen: Whether to exclude seen items
        user_interactions: Dict mapping user IDs to lists of seen item IDs
        
    Returns:
        Dict mapping user IDs to recommendation lists
    """
    if not self.is_fitted:
        raise ValueError("Model must be fitted before generating recommendations")
        
    if n is None:
        n = self.top_k
        
    # Process in batches to reduce memory usage
    batch_size = 100  # Adjust based on available memory
    all_recommendations = {}
    
    # Process users in batches
    for i in range(0, len(user_ids), batch_size):
        batch_users = user_ids[i:min(i+batch_size, len(user_ids))]
        
        for user_id in batch_users:
            # Get items this user has seen
            seen_items = user_interactions.get(user_id, []) if user_interactions and exclude_seen else None
            
            # Generate recommendations
            try:
                recs = self.recommend(user_id, n=n, exclude_seen=exclude_seen, seen_items=seen_items)
                all_recommendations[user_id] = recs
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_id}: {str(e)}")
                all_recommendations[user_id] = []
                
    return all_recommendations
```