"""
Content-based recipe recommender using TF-IDF and cosine similarity.

This module implements the ML core of the recommendation system.
It uses scikit-learn's TF-IDF vectorizer to compute ingredient similarity
and returns top-N recipe candidates based on cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional


class RecipeRecommender:
    """Content-based recipe recommendation engine."""
    
    def __init__(self, recipes_df: pd.DataFrame):
        """
        Initialize recommender with recipe dataset.
        
        Args:
            recipes_df: DataFrame with recipe data (must contain 'ingredients_cleaned')
        """
        self.recipes_df = recipes_df.copy()
        self.tfidf_vectorizer = None
        self.ingredient_tfidf_matrix = None
    
    def build_tfidf_model(self) -> None:
        """
        Build TF-IDF model on ingredient text.
        
        This vectorizes ingredient strings into numerical features
        suitable for cosine similarity computation.
        """
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words=None  # Keep all words (ingredient-specific)
        )
        
        # Fit and transform ingredient data
        self.ingredient_tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.recipes_df['ingredients_cleaned'].fillna('')
        )
    
    def compute_ingredient_similarity(
        self,
        user_ingredients: str,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Compute ingredient similarity between user input and recipes.
        
        Args:
            user_ingredients: User's ingredient input (comma-separated)
            top_n: Number of top matches to return
        
        Returns:
            DataFrame with recipe indices and similarity scores
        """
        if self.tfidf_vectorizer is None or self.ingredient_tfidf_matrix is None:
            raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")
        
        # Preprocess user input
        user_ingredients_cleaned = user_ingredients.lower().strip()
        
        # Transform user input using fitted vectorizer
        user_tfidf = self.tfidf_vectorizer.transform([user_ingredients_cleaned])
        
        # Compute cosine similarity
        similarities = cosine_similarity(user_tfidf, self.ingredient_tfidf_matrix)[0]
        
        # Get top-N matches
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_similarities = similarities[top_indices]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'recipe_index': top_indices,
            'ingredient_similarity': top_similarities
        })
        
        return result_df
    
    def recommend(
        self,
        user_ingredients: str,
        meal_type: Optional[str] = None,
        cuisine: Optional[str] = None,
        max_cooking_time: Optional[int] = None,
        n_recommendations: int = 5,
        scorer=None
    ) -> pd.DataFrame:
        """
        Generate recipe recommendations.
        
        This is the main entry point for recommendations. It:
        1. Computes content-based (ingredient) similarity
        2. Filters by hard constraints (cooking time, etc.)
        3. Uses preference scorer for weighted ranking
        4. Returns top-N recommendations
        
        Args:
            user_ingredients: Comma-separated ingredients
            meal_type: Optional meal type filter
            cuisine: Optional cuisine filter
            max_cooking_time: Optional max cooking time
            n_recommendations: Number of recipes to return
            scorer: PreferenceScorer instance (optional)
        
        Returns:
            DataFrame with top recommendations (sorted by final_score)
        """
        # Step 1: Get candidate recipes based on ingredient similarity
        candidates = self.compute_ingredient_similarity(
            user_ingredients=user_ingredients,
            top_n=len(self.recipes_df)  # Consider all initially
        )
        
        # Merge with recipe data
        result_df = self.recipes_df.iloc[candidates['recipe_index'].values].copy()
        result_df['ingredient_similarity'] = candidates['ingredient_similarity'].values
        
        # Step 2: Apply hard filters
        if max_cooking_time is not None:
            result_df = result_df[result_df['cooking_time'] <= max_cooking_time]
        
        if meal_type is not None:
            result_df = result_df[result_df['meal_type'] == meal_type]
        
        if cuisine is not None:
            result_df = result_df[result_df['cuisine'] == cuisine]
        
        # Step 3: Score and rank using preference scorer
        if scorer is not None:
            result_df['final_score'] = result_df.apply(
                lambda row: scorer.score(
                    ingredient_similarity=row['ingredient_similarity'],
                    cuisine_match=(1.0 if cuisine is None or row['cuisine'] == cuisine else 0.0),
                    meal_type_match=(1.0 if meal_type is None or row['meal_type'] == meal_type else 0.0),
                    cooking_time_score=self._compute_time_score(
                        row['cooking_time'],
                        max_cooking_time if max_cooking_time else 60
                    )
                ),
                axis=1
            )
        else:
            # If no scorer, use ingredient similarity as final score
            result_df['final_score'] = result_df['ingredient_similarity']
        
        # Step 4: Sort by final score and return top-N
        result_df = result_df.sort_values('final_score', ascending=False)
        
        return result_df[['recipe_name', 'ingredients', 'cuisine', 'meal_type',
                          'cooking_time', 'instructions', 'ingredient_similarity',
                          'final_score']].head(n_recommendations)
    
    @staticmethod
    def _compute_time_score(actual_time: float, max_time: float) -> float:
        """
        Compute cooking time score (penalty-based).
        
        Higher score for recipes within reasonable time.
        Lower score for recipes exceeding max_time.
        
        Args:
            actual_time: Actual cooking time
            max_time: Maximum acceptable cooking time
        
        Returns:
            Score between 0 and 1
        """
        if actual_time <= max_time:
            # Prefer shorter cooking times (but not penalize if within limit)
            return 0.9 + 0.1 * (1 - actual_time / max_time)
        else:
            # Penalty for exceeding max time
            excess_ratio = (actual_time - max_time) / max_time
            return max(0.0, 1.0 - excess_ratio * 0.5)
    
    def get_explainability_info(
        self,
        user_ingredients: str,
        recipe_index: int
    ) -> dict:
        """
        Provide explainability info for a recommendation.
        
        Args:
            user_ingredients: User's ingredient input
            recipe_index: Index of recommended recipe
        
        Returns:
            Dictionary with explanation details
        """
        recipe = self.recipes_df.iloc[recipe_index]
        
        # Compute similarity
        user_tfidf = self.tfidf_vectorizer.transform([user_ingredients.lower()])
        similarity = cosine_similarity(
            user_tfidf,
            self.ingredient_tfidf_matrix[recipe_index:recipe_index+1]
        )[0][0]
        
        return {
            'recipe_name': recipe['recipe_name'],
            'ingredient_similarity': similarity,
            'user_ingredients': user_ingredients,
            'recipe_ingredients': recipe['ingredients'],
            'cuisine': recipe['cuisine'],
            'meal_type': recipe['meal_type'],
            'cooking_time': recipe['cooking_time']
        }
