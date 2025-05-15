import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity

from models.base import BaseRecommender
from models.content_based import ContentBasedRecommender
from models.popular import PopularityRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class ColdStartRecommender(BaseRecommender):
    """
    Recommender specifically designed to handle cold-start users.

    This recommender uses a combination of strategies to provide recommendations
    for users with few or no interactions:
    1. Demographic similarity - recommend based on similar users' preferences
    2. Popular items in the user's demographic group
    3. Content-based recommendations using user profile attributes
    4. Diverse item selection to increase chances of relevance
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cold start recommender.

        Args:
            config: Dictionary with configuration parameters
        """
        super().__init__("ColdStart", config)

        # Component recommenders
        self.content_based = None
        self.popularity = None

        # Cold-start specific parameters
        self.min_interactions = config.get('models.cold_start.min_interactions', 5)
        self.diversity_weight = config.get('models.cold_start.diversity_weight', 0.3)
        self.demographic_fields = config.get('models.cold_start.demographic_fields',
                                             ['age_group', 'country', 'gender'])

        # Data containers
        self.user_features = None
        self.item_features = None
        self.interactions = None
        self.user_segments = None
        self.demographic_similarity = None
        self.segment_popularity = None

        logger.info(f"Initialized ColdStartRecommender with min_interactions={self.min_interactions}")

    def fit(self, train_data: pd.DataFrame, item_data: Optional[pd.DataFrame] = None,
            user_data: Optional[pd.DataFrame] = None,
            user_segments: Optional[Dict[str, List[int]]] = None) -> None:
        """
        Train the cold start recommender.

        Args:
            train_data: DataFrame with training data
            item_data: DataFrame with item features (optional)
            user_data: DataFrame with user features (optional)
            user_segments: Dictionary mapping segment names to lists of user IDs (optional)
        """
        logger.info("Fitting cold start recommender")

        # Get item and user data if not provided directly
        if item_data is None or user_data is None:
            # Try to extract from dataset if available
            from data.dataset import RetailDataset

            # Check if train_data has a dataset attribute
            dataset = getattr(train_data, 'dataset', None)
            if isinstance(dataset, RetailDataset):
                if item_data is None:
                    item_data = dataset.get_item_features()
                if user_data is None:
                    user_data = dataset.get_user_features()
            else:
                raise ValueError("Cold start recommender requires item_data and user_data")

        # Store data
        self.user_features = user_data
        self.item_features = item_data
        self.interactions = train_data
        self.user_segments = user_segments

        # Initialize component recommenders
        logger.info("Initializing component recommenders")

        self.content_based = ContentBasedRecommender(self.config)
        self.content_based.fit(train_data, item_data, user_data)

        self.popularity = PopularityRecommender(self.config)
        self.popularity.fit(train_data)

        # Set ID mappings for component recommenders
        if self.user_mapping is not None and self.item_mapping is not None:
            self.content_based.set_id_mappings(self.user_mapping, self.item_mapping)
            self.popularity.set_id_mappings(self.user_mapping, self.item_mapping)

        # Build demographic-based user similarities
        self._build_demographic_similarities()

        self.is_fitted = True
        logger.info("Cold start recommender fitted successfully")

    def _build_demographic_similarities(self) -> None:
        """
        Build demographic-based user similarities.
        """
        logger.info("Building demographic-based user similarities")

        # Prepare user features (use only demographic fields)
        available_fields = [field for field in self.demographic_fields if field in self.user_features.columns]

        if not available_fields:
            logger.warning("No demographic fields found in user features")
            # Create dummy similarity matrix (identity)
            self.demographic_similarity = np.eye(len(self.user_features))
            return

        demographic_features = self.user_features[available_fields].copy()

        # Convert categorical features to one-hot encoding
        categorical_cols = demographic_features.select_dtypes(include=['object']).columns

        if not categorical_cols.empty:
            one_hot = pd.get_dummies(
                demographic_features[categorical_cols],
                prefix=categorical_cols,
                dummy_na=False
            )
            demographic_features = demographic_features.drop(categorical_cols, axis=1)
            demographic_features = pd.concat([demographic_features, one_hot], axis=1)

        # Normalize numerical features
        numerical_cols = demographic_features.select_dtypes(include=['number']).columns

        for col in numerical_cols:
            min_val = demographic_features[col].min()
            max_val = demographic_features[col].max()

            if max_val > min_val:
                demographic_features[col] = (demographic_features[col] - min_val) / (max_val - min_val)

        # Calculate user-user similarity based on demographics
        self.demographic_similarity = cosine_similarity(demographic_features)

        # Build segment-based popularity
        self.segment_popularity = {}

        if self.user_segments:
            for segment, user_ids in self.user_segments.items():
                # Get interactions for users in this segment
                segment_interactions = self.interactions[self.interactions['customer_key'].isin(user_ids)]

                if segment_interactions.empty:
                    continue

                # Calculate popularity scores
                item_col = 'product_key'

                if 'rating' in segment_interactions.columns:
                    # Use sum of ratings as popularity
                    segment_popularity = segment_interactions.groupby(item_col)['rating'].sum().to_dict()
                else:
                    # Use interaction count as popularity
                    segment_popularity = segment_interactions[item_col].value_counts().to_dict()

                self.segment_popularity[segment] = segment_popularity

        logger.info("Demographic similarity and segment popularity built successfully")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Get user interaction count
        if user_id in self.interactions['customer_key'].values:
            user_interactions = self.interactions[self.interactions['customer_key'] == user_id]
            interaction_count = len(user_interactions)
        else:
            interaction_count = 0

        # Use different strategies based on interaction count
        if interaction_count >= self.min_interactions:
            # Use content-based recommender for users with some interactions
            return self.content_based.predict(user_id, item_id)
        else:
            # For cold-start users, use a combination of strategies

            # 1. Demographic similarity-based prediction
            demographic_score = self._predict_demographic_similarity(user_id, item_id)

            # 2. Segment popularity-based prediction
            segment_score = self._predict_segment_popularity(user_id, item_id)

            # 3. Content-based prediction (if possible)
            content_score = 0.0
            try:
                content_score = self.content_based.predict(user_id, item_id)
            except Exception as e:
                logger.debug(f"Content-based prediction failed: {str(e)}")

            # Combine scores (simple average for now)
            combined_score = (demographic_score + segment_score + content_score) / 3.0

            return combined_score

    def _predict_demographic_similarity(self, user_id: int, item_id: int) -> float:
        """
        Predict based on demographically similar users.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Prediction score
        """
        # Check if user exists in demographic data
        if user_id not in self.user_features.index:
            return 0.0

        user_idx = self.user_features.index.get_loc(user_id)

        # Get demographic similarities to other users
        similarities = self.demographic_similarity[user_idx]

        # Get users who have interacted with the item
        item_interactions = self.interactions[self.interactions['product_key'] == item_id]
        item_users = item_interactions['customer_key'].unique()

        # Convert to indices
        item_user_indices = []
        for u in item_users:
            if u in self.user_features.index:
                try:
                    item_user_indices.append(self.user_features.index.get_loc(u))
                except:
                    continue

        # Calculate weighted score
        if not item_user_indices:
            return 0.0

        weighted_sum = sum(similarities[idx] for idx in item_user_indices)
        return weighted_sum / len(item_user_indices)

    def _predict_segment_popularity(self, user_id: int, item_id: int) -> float:
        """
        Predict based on item popularity in the user's segment.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Prediction score
        """
        # Find user's segment
        user_segment = None

        if self.user_segments:
            for segment, user_ids in self.user_segments.items():
                if user_id in user_ids:
                    user_segment = segment
                    break

        # If user is in a segment, use segment popularity
        if user_segment and user_segment in self.segment_popularity:
            # Normalize the segment popularity score
            segment_items = self.segment_popularity[user_segment]
            max_score = max(segment_items.values()) if segment_items else 1.0

            return segment_items.get(item_id, 0.0) / max_score
        else:
            # Otherwise, use general popularity
            return self.popularity.predict(user_id, item_id)

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a cold-start user.

        Args:
            user_id: User ID
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            seen_items: List of item IDs the user has already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")

        if n is None:
            n = self.top_k

        # Get user interaction count
        user_interactions = self.interactions[self.interactions['customer_key'] == user_id]
        interaction_count = len(user_interactions)

        # Use different strategies based on interaction count
        if interaction_count >= self.min_interactions:
            # Use content-based recommender for users with some interactions
            return self.content_based.recommend(
                user_id, n=n, exclude_seen=exclude_seen, seen_items=seen_items
            )
        else:
            # For cold-start users, use a multi-strategy approach

            # 1. Get demographic-based recommendations
            demographic_recs = self._recommend_demographic_similarity(
                user_id, n=n * 2, exclude_seen=exclude_seen, seen_items=seen_items
            )

            # 2. Get segment popularity-based recommendations
            segment_recs = self._recommend_segment_popularity(
                user_id, n=n * 2, exclude_seen=exclude_seen, seen_items=seen_items
            )

            # 3. Get some content-based recommendations if possible
            content_recs = []
            try:
                content_recs = self.content_based.recommend(
                    user_id, n=n * 2, exclude_seen=exclude_seen, seen_items=seen_items
                )
            except Exception as e:
                logger.debug(f"Content-based recommendations failed: {str(e)}")

            # Combine recommendations with score normalization
            combined_recs = self._combine_recommendations([
                (demographic_recs, 0.4),
                (segment_recs, 0.4),
                (content_recs, 0.2)
            ])

            # Apply diversity enhancement
            recommendations = self._enhance_diversity(
                combined_recs, n=n, diversity_weight=self.diversity_weight
            )

        # Return top n recommendations
        return recommendations[:n]

    def _recommend_demographic_similarity(self, user_id: int, n: int = 10,
                                          exclude_seen: bool = True,
                                          seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations based on demographically similar users.

        Args:
            user_id: User ID
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            seen_items: List of item IDs the user has already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        # Check if user exists in demographic data
        if user_id not in self.user_features.index:
            return []

        user_idx = self.user_features.index.get_loc(user_id)

        # Get demographic similarities to other users
        similarities = self.demographic_similarity[user_idx]

        # Get top similar users
        top_n = min(50, len(similarities))
        top_user_indices = np.argsort(similarities)[::-1][:top_n]  # Get top similar users

        # Get items these users have interacted with
        item_scores = {}

        for idx in top_user_indices:
            if idx == user_idx:  # Skip the user itself
                continue

            # Get original user ID
            similar_user_id = self.user_features.index[idx]

            # Get items this user has interacted with
            user_items = self.interactions[self.interactions['customer_key'] == similar_user_id]['product_key'].unique()

            # Add weighted scores
            for item in user_items:
                if exclude_seen and seen_items and item in seen_items:
                    continue

                if item not in item_scores:
                    item_scores[item] = 0.0

                item_scores[item] += similarities[idx]

        # Sort by score
        recommendations = [(item, score) for item, score in item_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n]

    def _recommend_segment_popularity(self, user_id: int, n: int = 10,
                                      exclude_seen: bool = True,
                                      seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations based on popularity in the user's segment.

        Args:
            user_id: User ID
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            seen_items: List of item IDs the user has already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        # Find user's segment
        user_segment = None

        if self.user_segments:
            for segment, user_ids in self.user_segments.items():
                if user_id in user_ids:
                    user_segment = segment
                    break

        # If user is in a segment, use segment popularity
        if user_segment and user_segment in self.segment_popularity:
            # Get popular items in the segment
            segment_items = self.segment_popularity[user_segment]

            # Filter out seen items if requested
            if exclude_seen and seen_items:
                segment_items = {item: score for item, score in segment_items.items()
                                 if item not in seen_items}

            # Sort by popularity
            recommendations = [(item, score) for item, score in segment_items.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)

            return recommendations[:n]
        else:
            # Otherwise, use general popularity
            return self.popularity.recommend(
                user_id, n=n, exclude_seen=exclude_seen, seen_items=seen_items
            )

    def _combine_recommendations(self, rec_lists_with_weights: List[Tuple[List[Tuple[int, float]], float]]) -> List[
        Tuple[int, float]]:
        """
        Combine multiple recommendation lists with weights.

        Args:
            rec_lists_with_weights: List of (recommendations, weight) tuples

        Returns:
            Combined and normalized list of (item_id, score) tuples
        """
        # Normalize scores within each list
        normalized_recs = []

        for recs, weight in rec_lists_with_weights:
            if not recs:
                continue

            # Find max score for normalization
            max_score = max(score for _, score in recs)

            if max_score > 0:
                # Normalize and apply weight
                norm_recs = [(item, (score / max_score) * weight) for item, score in recs]
                normalized_recs.append(norm_recs)
            else:
                normalized_recs.append([(item, 0.0) for item, _ in recs])

        # Combine all recommendations
        combined_scores = {}

        for recs in normalized_recs:
            for item, score in recs:
                if item not in combined_scores:
                    combined_scores[item] = 0.0

                combined_scores[item] += score

        # Create final recommendations list
        combined_recs = [(item, score) for item, score in combined_scores.items()]
        combined_recs.sort(key=lambda x: x[1], reverse=True)

        return combined_recs

    def _enhance_diversity(self, recommendations: List[Tuple[int, float]], n: int = 10,
                           diversity_weight: float = 0.3) -> List[Tuple[int, float]]:
        """
        Enhance diversity in the recommendations list.

        Args:
            recommendations: List of (item_id, score) tuples
            n: Number of recommendations to generate
            diversity_weight: Weight for diversity (0.0 to 1.0)

        Returns:
            Diversified list of (item_id, score) tuples
        """
        if not recommendations or len(recommendations) <= n:
            return recommendations

        if diversity_weight <= 0.0:
            return recommendations[:n]

        if diversity_weight >= 1.0:
            return self._maximal_marginal_relevance(recommendations, n)

        # Calculate how many recommendations to take from each approach
        direct_count = int(n * (1.0 - diversity_weight))
        diverse_count = n - direct_count

        # Get top recommendations directly by score
        direct_recs = recommendations[:direct_count]

        # Get diverse recommendations using maximal marginal relevance
        selected_items = [item for item, _ in direct_recs]
        remaining_recs = [(item, score) for item, score in recommendations
                          if item not in selected_items]

        diverse_recs = self._maximal_marginal_relevance(remaining_recs, diverse_count, selected_items)

        # Combine and return
        final_recs = direct_recs + diverse_recs
        return final_recs

    def _maximal_marginal_relevance(self, recommendations: List[Tuple[int, float]], n: int = 10,
                                    selected_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Implement maximal marginal relevance for diverse recommendation.

        Args:
            recommendations: List of (item_id, score) tuples
            n: Number of items to select
            selected_items: Items already selected

        Returns:
            Diversified list of (item_id, score) tuples
        """
        if not recommendations:
            return []

        if n >= len(recommendations):
            return recommendations

        # Get item features for diversity calculation
        item_features = self.item_features.copy()

        # Initialize selected items
        if selected_items is None:
            selected_items = []
            selected_recs = []
        else:
            selected_recs = [(item, 0.0) for item in selected_items]  # Dummy scores

        # Calculate similarity between items
        # Select available items and compute their feature matrix
        available_items = [item for item, _ in recommendations]
        item_indices = []

        for item in available_items:
            if item in item_features.index:
                try:
                    item_indices.append(item_features.index.get_loc(item))
                except:
                    continue

        if not item_indices:
            return recommendations[:n]

        # Get item features for similarity calculation
        item_feature_matrix = item_features.iloc[item_indices].select_dtypes(include=['number'])

        # If no numerical features, use dummy features
        if item_feature_matrix.empty:
            return recommendations[:n]

        # Compute item-item similarity
        item_similarity = cosine_similarity(item_feature_matrix)

        # Get mapping from item ID to matrix index
        item_to_idx = {item: i for i, item in enumerate(available_items) if i < len(item_indices)}

        # Implement MMR selection
        lambda_param = 0.5  # Balance between relevance and diversity

        while len(selected_recs) < n + len(selected_items) and recommendations:
            max_mmr = -float('inf')
            max_idx = -1

            for i, (item, score) in enumerate(recommendations):
                # Skip if item already selected
                if item in selected_items:
                    continue

                # Skip if item not in similarity matrix
                if item not in item_to_idx:
                    continue

                # Calculate similarity to already selected items
                sim_to_selected = 0.0

                if selected_items:
                    # Calculate max similarity to any selected item
                    for sel_item in selected_items:
                        if sel_item in item_to_idx:
                            idx1 = item_to_idx[item]
                            idx2 = item_to_idx[sel_item]
                            if idx1 < len(item_similarity) and idx2 < len(item_similarity):
                                sim_to_selected = max(sim_to_selected, item_similarity[idx1][idx2])

                # Calculate MMR score
                mmr = lambda_param * score - (1.0 - lambda_param) * sim_to_selected

                if mmr > max_mmr:
                    max_mmr = mmr
                    max_idx = i

            if max_idx == -1:
                break

            # Add to selected items
            item, score = recommendations[max_idx]
            selected_items.append(item)
            selected_recs.append((item, score))

            # Remove from candidates
            recommendations.pop(max_idx)

        # Return only the newly selected items
        return selected_recs[len(selected_items) - min(n, len(selected_recs)):]

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data
        """
        return {
            'min_interactions': self.min_interactions,
            'diversity_weight': self.diversity_weight,
            'demographic_fields': self.demographic_fields,
            'content_based': self.content_based._get_model_specific_data() if self.content_based else None,
            'popularity': self.popularity._get_model_specific_data() if self.popularity else None,
            'demographic_similarity': self.demographic_similarity.tolist() if hasattr(self,
                                                                                      'demographic_similarity') else None,
            'segment_popularity': self.segment_popularity if hasattr(self, 'segment_popularity') else None
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data
        """
        self.min_interactions = data['min_interactions']
        self.diversity_weight = data['diversity_weight']
        self.demographic_fields = data['demographic_fields']

        if data['content_based']:
            from models.content_based import ContentBasedRecommender
            self.content_based = ContentBasedRecommender(self.config)
            self.content_based._restore_model_specific_data(data['content_based'])

        if data['popularity']:
            from models.popular import PopularityRecommender
            self.popularity = PopularityRecommender(self.config)
            self.popularity._restore_model_specific_data(data['popularity'])

        if data['demographic_similarity']:
            self.demographic_similarity = np.array(data['demographic_similarity'])

        if data['segment_popularity']:
            self.segment_popularity = data['segment_popularity']