"""
Preference-aware scoring engine.

Combines multiple preference signals into a final recommendation score:
- Ingredient similarity (highest weight - 60%)
- Cuisine match (medium weight - 20%)
- Meal type match (medium weight - 10%)
- Cooking time suitability (10%)

Implements hard filters and soft preference boosts.
"""

import numpy as np
from typing import Optional


class PreferenceScorer:
    """Weighted preference scoring engine."""
    
    # Configurable weights
    WEIGHTS = {
        'ingredient_similarity': 0.60,
        'cuisine_match': 0.20,
        'meal_type_match': 0.10,
        'cooking_time': 0.10
    }
    
    def __init__(
        self,
        weights: Optional[dict] = None
    ):
        """
        Initialize scorer with configurable weights.
        
        Args:
            weights: Optional dictionary to override default weights
        """
        if weights is not None:
            self.WEIGHTS.update(weights)
        
        # Validate weights sum to 1.0
        total_weight = sum(self.WEIGHTS.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def score(
        self,
        ingredient_similarity: float,
        cuisine_match: float,
        meal_type_match: float,
        cooking_time_score: float
    ) -> float:
        """
        Compute final preference score.
        
        Args:
            ingredient_similarity: Ingredient TF-IDF similarity (0-1)
            cuisine_match: Cuisine match indicator (0-1)
            meal_type_match: Meal type match indicator (0-1)
            cooking_time_score: Time suitability score (0-1)
        
        Returns:
            Final weighted score (0-1)
        """
        # Ensure inputs are bounded
        ingredient_similarity = np.clip(ingredient_similarity, 0, 1)
        cuisine_match = np.clip(cuisine_match, 0, 1)
        meal_type_match = np.clip(meal_type_match, 0, 1)
        cooking_time_score = np.clip(cooking_time_score, 0, 1)
        
        # Compute weighted sum
        final_score = (
            self.WEIGHTS['ingredient_similarity'] * ingredient_similarity +
            self.WEIGHTS['cuisine_match'] * cuisine_match +
            self.WEIGHTS['meal_type_match'] * meal_type_match +
            self.WEIGHTS['cooking_time'] * cooking_time_score
        )
        
        return final_score
    
    def get_score_breakdown(
        self,
        ingredient_similarity: float,
        cuisine_match: float,
        meal_type_match: float,
        cooking_time_score: float
    ) -> dict:
        """
        Get detailed score breakdown for explainability.
        
        Args:
            ingredient_similarity: Ingredient similarity score
            cuisine_match: Cuisine match score
            meal_type_match: Meal type match score
            cooking_time_score: Cooking time score
        
        Returns:
            Dictionary with individual component scores and final score
        """
        return {
            'ingredient_similarity': {
                'score': ingredient_similarity,
                'weight': self.WEIGHTS['ingredient_similarity'],
                'contribution': ingredient_similarity * self.WEIGHTS['ingredient_similarity']
            },
            'cuisine_match': {
                'score': cuisine_match,
                'weight': self.WEIGHTS['cuisine_match'],
                'contribution': cuisine_match * self.WEIGHTS['cuisine_match']
            },
            'meal_type_match': {
                'score': meal_type_match,
                'weight': self.WEIGHTS['meal_type_match'],
                'contribution': meal_type_match * self.WEIGHTS['meal_type_match']
            },
            'cooking_time': {
                'score': cooking_time_score,
                'weight': self.WEIGHTS['cooking_time'],
                'contribution': cooking_time_score * self.WEIGHTS['cooking_time']
            },
            'final_score': self.score(
                ingredient_similarity,
                cuisine_match,
                meal_type_match,
                cooking_time_score
            )
        }
