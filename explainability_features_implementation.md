# Explainability Features Implementation Plan

## 1. Implement Recommendation Explanation Module

### Implementation Steps:

1. **Create a new module for explainability**:

```python
# models/explainability.py
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from collections import defaultdict
import json

class RecommendationExplainer:
    """
    Provides explanation features for recommendations from various models.
    """
    
    def __init__(self, item_features: pd.DataFrame, 
                 user_features: Optional[pd.DataFrame] = None,
                 item_similarity: Optional[np.ndarray] = None,
                 user_item_interactions: Optional[pd.DataFrame] = None):
        """
        Initialize the explainer with necessary data.
        
        Args:
            item_features: DataFrame with item features
            user_features: Optional DataFrame with user features
            item_similarity: Optional item-item similarity matrix
            user_item_interactions: Optional DataFrame with user-item interactions
        """
        self.item_features = item_features
        self.user_features = user_features
        self.item_similarity = item_similarity
        self.user_item_interactions = user_item_interactions
        
        # Extract feature names
        self.categorical_features = item_features.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = item_features.select_dtypes(include=['number']).columns.tolist()
        
    def explain_content_based(self, user_id: int, item_id: int, 
                              user_profile: np.ndarray, 
                              top_n_features: int = 3) -> Dict[str, Any]:
        """
        Explain a content-based recommendation.
        
        Args:
            user_id: User ID
            item_id: Recommended item ID
            user_profile: User's content profile vector
            top_n_features: Number of top features to include
            
        Returns:
            Dictionary with explanation details
        """
        if item_id not in self.item_features.index:
            return {"error": f"Item {item_id} not found in features"}
            
        # Get item features
        item_features = self.item_features.loc[item_id]
        
        # Find most important categorical features
        cat_explanations = {}
        for feature in self.categorical_features:
            if pd.notna(item_features[feature]):
                cat_explanations[feature] = item_features[feature]
                
        # Find most important numerical features by comparing to user profile
        num_explanations = {}
        for feature in self.numerical_features:
            if feature in item_features and pd.notna(item_features[feature]):
                # Calculate relevance score (placeholder - actual implementation depends on user profile structure)
                relevance = 1.0  # Default value
                num_explanations[feature] = {
                    "value": item_features[feature],
                    "relevance": relevance
                }
                
        # Get similar items the user has interacted with
        similar_items = []
        if self.item_similarity is not None and self.user_item_interactions is not None:
            # Get items user has interacted with
            user_items = self.user_item_interactions[
                self.user_item_interactions['customer_key'] == user_id
            ]['product_key'].unique()
            
            # Get similarity scores for the recommended item
            item_idx = self.item_features.index.get_loc(item_id)
            for other_item in user_items:
                if other_item in self.item_features.index:
                    other_idx = self.item_features.index.get_loc(other_item)
                    similarity = self.item_similarity[item_idx, other_idx]
                    similar_items.append((other_item, similarity))
                    
            # Sort by similarity
            similar_items.sort(key=lambda x: x[1], reverse=True)
            
        # Create explanation
        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "item_name": item_features.get('product_name', f"Item {item_id}"),
            "categorical_features": cat_explanations,
            "numerical_features": num_explanations,
            "similar_items_liked": [
                {"item_id": item, "similarity": sim} 
                for item, sim in similar_items[:3]
            ] if similar_items else []
        }
        
        return explanation
        
    def explain_collaborative(self, user_id: int, item_id: int, 
                             similar_users: List[Tuple[int, float]] = None,
                             similar_items: List[Tuple[int, float]] = None) -> Dict[str, Any]:
        """
        Explain a collaborative filtering recommendation.
        
        Args:
            user_id: User ID
            item_id: Recommended item ID
            similar_users: List of (user_id, similarity) tuples for user-based CF
            similar_items: List of (item_id, similarity) tuples for item-based CF
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "item_name": self.item_features.loc[item_id].get('product_name', f"Item {item_id}") 
                          if item_id in self.item_features.index else f"Item {item_id}"
        }
        
        # User-based explanation
        if similar_users:
            explanation["similar_users"] = [
                {"user_id": user, "similarity": sim} 
                for user, sim in similar_users[:3]
            ]
            
        # Item-based explanation
        if similar_items:
            # Get item names if available
            similar_items_with_names = []
            for item, sim in similar_items[:3]:
                name = self.item_features.loc[item].get('product_name', f"Item {item}") \
                       if item in self.item_features.index else f"Item {item}"
                similar_items_with_names.append({
                    "item_id": item,
                    "name": name,
                    "similarity": sim
                })
                
            explanation["similar_items"] = similar_items_with_names
            
        # Find common features if similar items exist
        if similar_items and self.item_features is not None:
            common_features = self._extract_common_features(item_id, [i for i, _ in similar_items[:3]])
            if common_features:
                explanation["common_features"] = common_features
                
        return explanation
        
    def explain_hybrid(self, user_id: int, item_id: int, 
                      component_scores: Dict[str, float],
                      component_explanations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Explain a hybrid recommendation.
        
        Args:
            user_id: User ID
            item_id: Recommended item ID
            component_scores: Scores from each component model
            component_explanations: Explanations from each component model
            
        Returns:
            Dictionary with explanation details
        """
        # Calculate contribution percentage from each component
        total_score = sum(component_scores.values())
        contributions = {}
        
        if total_score > 0:
            for component, score in component_scores.items():
                contributions[component] = (score / total_score) * 100
                
        # Create explanation
        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "item_name": self.item_features.loc[item_id].get('product_name', f"Item {item_id}") 
                          if item_id in self.item_features.index else f"Item {item_id}",
            "component_contributions": contributions,
            "component_explanations": component_explanations
        }
        
        return explanation
        
    def _extract_common_features(self, target_item_id: int, similar_item_ids: List[int]) -> Dict[str, Any]:
        """
        Extract common features between target item and similar items.
        
        Args:
            target_item_id: Target item ID
            similar_item_ids: List of similar item IDs
            
        Returns:
            Dictionary with common feature information
        """
        if target_item_id not in self.item_features.index:
            return {}
            
        common_features = {}
        
        # Get target item features
        target_features = self.item_features.loc[target_item_id]
        
        # Check each categorical feature
        for feature in self.categorical_features:
            if pd.notna(target_features[feature]):
                target_value = target_features[feature]
                matching_items = []
                
                for item_id in similar_item_ids:
                    if item_id in self.item_features.index:
                        item_features = self.item_features.loc[item_id]
                        if pd.notna(item_features[feature]) and item_features[feature] == target_value:
                            matching_items.append(item_id)
                            
                if matching_items:
                    common_features[feature] = {
                        "value": target_value,
                        "matching_items": matching_items,
                        "match_ratio": len(matching_items) / len(similar_item_ids)
                    }
                    
        return common_features
```

2. **Modify your recommender models to include explanations**:

```python
# Add to models/content_based.py

def predict_with_explanation(self, user_id: int, item_id: int) -> Tuple[float, Dict[str, Any]]:
    """
    Predict rating with explanation.
    
    Args:
        user_id: User ID
        item_id: Item ID
        
    Returns:
        Tuple of (predicted rating, explanation dict)
    """
    from models.explainability import RecommendationExplainer
    
    # Get prediction
    prediction = self.predict(user_id, item_id)
    
    # Create explainer
    explainer = RecommendationExplainer(
        item_features=self.item_features,
        item_similarity=self.item_similarity_matrix
    )
    
    # Get user profile
    user_profile = self.user_profiles.get(user_id, None)
    
    # Generate explanation
    explanation = explainer.explain_content_based(
        user_id=user_id,
        item_id=item_id,
        user_profile=user_profile
    )
    
    return prediction, explanation

def recommend_with_explanations(self, user_id: int, n: int = None,
                               exclude_seen: bool = True,
                               seen_items: Optional[List[int]] = None) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Generate recommendations with explanations.
    
    Args:
        user_id: User ID
        n: Number of recommendations
        exclude_seen: Whether to exclude seen items
        seen_items: List of seen items
        
    Returns:
        List of (item_id, score, explanation) tuples
    """
    # Get recommendations
    recommendations = self.recommend(user_id, n, exclude_seen, seen_items)
    
    # Add explanations
    explained_recommendations = []
    for item_id, score in recommendations:
        _, explanation = self.predict_with_explanation(user_id, item_id)
        explained_recommendations.append((item_id, score, explanation))
        
    return explained_recommendations
```

3. **Add similar methods to your collaborative and hybrid models**

## 2. Add Confidence Scores

### Implementation Steps:

1. **Create a confidence score module**:

```python
# models/confidence.py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

class ConfidenceCalculator:
    """
    Calculate confidence scores for recommendations.
    """
    
    @staticmethod
    def calculate_content_confidence(user_id: int, 
                                    item_id: int,
                                    user_profile: np.ndarray,
                                    item_features: np.ndarray,
                                    user_history_size: int,
                                    prediction_score: float) -> float:
        """
        Calculate confidence for content-based recommendation.
        
        Args:
            user_id: User ID
            item_id: Item ID
            user_profile: User's content profile
            item_features: Item's feature vector
            user_history_size: Number of items user has interacted with
            prediction_score: Predicted score
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence starts with prediction score
        base_confidence = min(1.0, prediction_score / 5.0)
        
        # Adjust based on user history - more history = more confidence
        history_factor = min(1.0, user_history_size / 10.0)
        
        # Calculate overall confidence
        confidence = 0.7 * base_confidence + 0.3 * history_factor
        
        return confidence
        
    @staticmethod
    def calculate_collaborative_confidence(user_id: int,
                                          item_id: int,
                                          num_neighbors: int,
                                          similarity_scores: List[float],
                                          variance: float,
                                          prediction_score: float) -> float:
        """
        Calculate confidence for collaborative recommendation.
        
        Args:
            user_id: User ID
            item_id: Item ID
            num_neighbors: Number of neighbors used
            similarity_scores: Similarity scores of neighbors
            variance: Variance in ratings from neighbors
            prediction_score: Predicted score
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence starts with prediction score
        base_confidence = min(1.0, prediction_score / 5.0)
        
        # More neighbors and higher similarities = more confidence
        neighbor_factor = min(1.0, num_neighbors / 10.0)
        similarity_factor = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Lower variance = more confidence
        variance_factor = 1.0 - min(1.0, variance)
        
        # Calculate overall confidence
        confidence = 0.4 * base_confidence + 0.3 * neighbor_factor + \
                     0.2 * similarity_factor + 0.1 * variance_factor
                     
        return confidence
        
    @staticmethod
    def calculate_hybrid_confidence(component_confidences: Dict[str, float],
                                  component_weights: Dict[str, float]) -> float:
        """
        Calculate confidence for hybrid recommendation.
        
        Args:
            component_confidences: Confidences from component models
            component_weights: Weights of component models
            
        Returns:
            Confidence score between 0 and 1
        """
        # Weighted average of component confidences
        confidence = 0.0
        total_weight = 0.0
        
        for component, conf in component_confidences.items():
            weight = component_weights.get(component, 0.0)
            confidence += conf * weight
            total_weight += weight
            
        if total_weight > 0:
            confidence /= total_weight
            
        return confidence
```

2. **Add confidence score calculation to your recommenders**:

```python
# Add to models/content_based.py

def calculate_confidence(self, user_id: int, item_id: int, prediction: float) -> float:
    """
    Calculate confidence score for a prediction.
    
    Args:
        user_id: User ID
        item_id: Item ID
        prediction: Predicted rating
        
    Returns:
        Confidence score between 0 and 1
    """
    from models.confidence import ConfidenceCalculator
    
    # Get user profile and item features
    user_profile = self.user_profiles.get(user_id, None)
    item_features = None
    
    if item_id in self.item_features.index:
        item_idx = self.item_features.index.get_loc(item_id)
        item_features = self.item_similarity_matrix[item_idx]
    
    # Get user history size
    user_history_size = 0
    if hasattr(self, 'interactions') and self.interactions is not None:
        user_history = self.interactions[self.interactions['customer_key'] == user_id]
        user_history_size = len(user_history)
    
    # Calculate confidence
    confidence = ConfidenceCalculator.calculate_content_confidence(
        user_id=user_id,
        item_id=item_id,
        user_profile=user_profile,
        item_features=item_features,
        user_history_size=user_history_size,
        prediction_score=prediction
    )
    
    return confidence

def recommend_with_confidence(self, user_id: int, n: int = None,
                             exclude_seen: bool = True,
                             seen_items: Optional[List[int]] = None) -> List[Tuple[int, float, float]]:
    """
    Generate recommendations with confidence scores.
    
    Args:
        user_id: User ID
        n: Number of recommendations
        exclude_seen: Whether to exclude seen items
        seen_items: List of seen items
        
    Returns:
        List of (item_id, score, confidence) tuples
    """
    # Get recommendations
    recommendations = self.recommend(user_id, n, exclude_seen, seen_items)
    
    # Add confidence scores
    confident_recommendations = []
    for item_id, score in recommendations:
        confidence = self.calculate_confidence(user_id, item_id, score)
        confident_recommendations.append((item_id, score, confidence))
        
    return confident_recommendations
```

3. **Combine explainability and confidence in a single method**:

```python
def recommend_with_details(self, user_id: int, n: int = None,
                         exclude_seen: bool = True,
                         seen_items: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Generate detailed recommendations with scores, confidence, and explanations.
    
    Args:
        user_id: User ID
        n: Number of recommendations
        exclude_seen: Whether to exclude seen items
        seen_items: List of seen items
        
    Returns:
        List of recommendation detail dictionaries
    """
    # Get recommendations
    recommendations = self.recommend(user_id, n, exclude_seen, seen_items)
    
    # Add details
    detailed_recommendations = []
    for item_id, score in recommendations:
        # Calculate confidence
        confidence = self.calculate_confidence(user_id, item_id, score)
        
        # Get explanation
        _, explanation = self.predict_with_explanation(user_id, item_id)
        
        # Create detailed recommendation
        detailed_rec = {
            "item_id": item_id,
            "score": score,
            "confidence": confidence,
            "explanation": explanation
        }
        
        # Add item name if available
        if item_id in self.item_features.index:
            item_name = self.item_features.loc[item_id].get('product_name', f"Item {item_id}")
            detailed_rec["item_name"] = item_name
            
        detailed_recommendations.append(detailed_rec)
        
    return detailed_recommendations
```