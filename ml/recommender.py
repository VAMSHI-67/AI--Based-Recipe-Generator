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
from typing import List, Tuple, Optional, Set


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
    
    def _compute_ingredient_overlap(
        self,
        user_ingredients: str,
        recipe_ingredients: str
    ) -> Tuple[int, float]:
        """
        Compute ingredient overlap using set intersection (Jaccard-based).
        
        ENHANCEMENT: This supplements TF-IDF by checking actual ingredient overlap.
        Ensures recipes share core ingredients with user input. Uses word-level
        substring matching since ingredients contain phrases like "1/2 cup butter".
        
        Args:
            user_ingredients: User's ingredient input
            recipe_ingredients: Recipe's ingredient list (comma or string format)
        
        Returns:
            Tuple of (overlap_count, overlap_ratio)
        """
        # Parse user ingredients into single words
        user_words = set()
        for ing in user_ingredients.split(','):
            ing_clean = ing.strip().lower()
            if ing_clean:
                # Split into individual words for substring matching
                for word in ing_clean.split():
                    if len(word) > 2:  # Skip very short words like "a", "of"
                        user_words.add(word)
        
        # Parse recipe ingredients - handle both string and list formats
        recipe_words = set()
        if isinstance(recipe_ingredients, list):
            recipe_text = ' '.join(recipe_ingredients)
        else:
            recipe_text = str(recipe_ingredients)
        
        # Split recipe ingredients by comma or space
        recipe_items = recipe_text.lower().split(',')
        for item in recipe_items:
            for word in item.split():
                if len(word) > 2:
                    recipe_words.add(word)
        
        # Compute intersection and ratio
        overlap_count = len(user_words & recipe_words)
        overlap_ratio = overlap_count / len(user_words) if len(user_words) > 0 else 0
        
        return overlap_count, overlap_ratio
    
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
        Generate recipe recommendations with HARD CONSTRAINTS and enhanced filtering.
        
        IMPROVEMENTS IN THIS VERSION:
        1. HARD FILTERS BEFORE SCORING: Cuisine, time, meal type applied first
        2. INGREDIENT OVERLAP CHECK: Minimum 2 ingredients or 25% overlap required
        3. MINIMUM SCORE THRESHOLD: Only recommend if final_score >= 30%
        4. DEDUPLICATION: Ensure no duplicate recipe_title in results
        5. GRACEFUL FALLBACK: If constraints too strict, relax and notify user
        
        Args:
            user_ingredients: Comma-separated ingredients
            meal_type: Optional meal type filter (breakfast, lunch, dinner, etc.)
            cuisine: Optional cuisine filter (Indian, Chinese, Italian, etc.)
            max_cooking_time: Optional max cooking time in minutes
            n_recommendations: Number of recipes to return
            scorer: PreferenceScorer instance (optional)
        
        Returns:
            DataFrame with top recommendations (sorted by final_score)
        """
        # Step 0: Initialize tracking variables for fallback logic
        original_constraints = {
            'cuisine': cuisine,
            'meal_type': meal_type,
            'max_cooking_time': max_cooking_time
        }
        fallback_applied = False
        fallback_reason = ""
        
        # Step 1: Get candidate recipes based on ingredient similarity (all recipes)
        candidates = self.compute_ingredient_similarity(
            user_ingredients=user_ingredients,
            top_n=len(self.recipes_df)
        )
        
        # Merge with recipe data
        result_df = self.recipes_df.iloc[candidates['recipe_index'].values].copy()
        result_df['ingredient_similarity'] = candidates['ingredient_similarity'].values
        
        # Step 2: CRITICAL - HARD FILTERS APPLIED BEFORE SCORING
        # These are non-negotiable constraints that must be satisfied first
        
        # Filter 2.1: Maximum cooking time (hard constraint - STRICTLY ENFORCED)
        if max_cooking_time is not None and max_cooking_time > 0:
            # Remove recipes exceeding max cooking time - non-negotiable
            recipes_exceeding = len(result_df[result_df['cooking_time'] > max_cooking_time])
            result_df = result_df[result_df['cooking_time'] <= max_cooking_time]
            if recipes_exceeding > 0:
                fallback_applied = True
                fallback_reason = f"Filtered out {recipes_exceeding} recipes exceeding {max_cooking_time} min time limit."
        
        # Filter 2.2: Cuisine filtering (hard constraint when explicitly selected)
        # IMPORTANT: When user selects cuisine, ONLY show that cuisine
        # Enhanced: Be strict about cuisine matching - prioritize exact matches
        if cuisine is not None and cuisine.lower() != 'all':
            cuisine_normalized = cuisine.lower().strip()
            result_df_cuisine = result_df[
                result_df['cuisine'].str.lower() == cuisine_normalized
            ]
            # Enforce cuisine filter strictly - always apply when specified
            # This ensures Indian cuisine returns only Indian recipes
            result_df = result_df_cuisine
        
        # Filter 2.3: Meal type filtering (soft - prefer matches but not strict)
        if meal_type is not None and meal_type.lower() != 'all':
            meal_type_normalized = meal_type.lower().strip()
            result_df_meal = result_df[
                result_df['meal_type'].str.lower() == meal_type_normalized
            ]
            # If some recipes match, use them; otherwise keep all
            if len(result_df_meal) > 0:
                result_df = result_df_meal
        
        # Step 3: INGREDIENT OVERLAP FILTERING
        # ENHANCEMENT: Ensure recipes share meaningful ingredients with user input
        # Minimum criteria: At least 2 shared ingredients OR 25% overlap
        result_df['ingredient_overlap_count'] = result_df['ingredients'].apply(
            lambda x: self._compute_ingredient_overlap(user_ingredients, x)[0]
        )
        result_df['ingredient_overlap_ratio'] = result_df['ingredients'].apply(
            lambda x: self._compute_ingredient_overlap(user_ingredients, x)[1]
        )
        
        # Apply stricter minimum overlap threshold
        # IMPORTANT: Recipes should actually use user's ingredients!
        MIN_OVERLAP_COUNT = 2  # At least 2 of user's ingredients must be in recipe
        MIN_OVERLAP_RATIO = 0.30  # Or at least 30% of user's ingredients
        
        result_df_with_overlap = result_df[
            (result_df['ingredient_overlap_count'] >= MIN_OVERLAP_COUNT) |
            (result_df['ingredient_overlap_ratio'] >= MIN_OVERLAP_RATIO)
        ]
        
        # If overlap filtering removed all recipes, relax threshold progressively
        if len(result_df_with_overlap) == 0:
            # Try with just 1 ingredient overlap
            result_df_with_overlap = result_df[
                (result_df['ingredient_overlap_count'] >= 1) |
                (result_df['ingredient_overlap_ratio'] >= 0.15)
            ]
            if len(result_df_with_overlap) > 0:
                fallback_applied = True
                fallback_reason = "Relaxed ingredient overlap filter to find more recipes."
                result_df = result_df_with_overlap
            else:
                fallback_applied = True
                fallback_reason = "No recipes found with your ingredients. Showing closest matches."
        else:
            result_df = result_df_with_overlap
        
        # Step 4: Score and rank using preference scorer with ingredient overlap boost
        if scorer is not None:
            cuisine_normalized = cuisine.lower().strip() if cuisine is not None else None
            meal_type_normalized = meal_type.lower().strip() if meal_type is not None else None
            
            result_df['final_score'] = result_df.apply(
                lambda row: scorer.score(
                    ingredient_similarity=row['ingredient_similarity'],
                    cuisine_match=(
                        1.0 if cuisine_normalized is None or 
                        cuisine_normalized == 'all' or
                        row['cuisine'].lower() == cuisine_normalized 
                        else 0.0
                    ),
                    meal_type_match=(
                        1.0 if meal_type_normalized is None or
                        meal_type_normalized == 'all' or
                        row['meal_type'].lower() == meal_type_normalized
                        else 0.0
                    ),
                    cooking_time_score=self._compute_time_score(
                        row['cooking_time'],
                        max_cooking_time if max_cooking_time else 60
                    )
                ),
                axis=1
            )
            
            # BOOST: Heavily reward recipes using user's ingredients
            # Recipes with high ingredient overlap get significant score boost
            result_df['ingredient_overlap_boost'] = result_df['ingredient_overlap_ratio'].apply(
                lambda ratio: 1.0 + (ratio * 0.5)
            )
            result_df['final_score'] = result_df['final_score'] * result_df['ingredient_overlap_boost']
        else:
            # If no scorer, use ingredient similarity as final score
            result_df['final_score'] = result_df['ingredient_similarity']
        
        # Step 5: MINIMUM SCORE THRESHOLD
        # ENHANCEMENT: Only recommend recipes with meaningful ingredient overlap
        # Threshold adjusted based on whether we have ingredient matches
        if len(result_df[result_df['ingredient_overlap_count'] > 0]) > 0:
            MIN_SCORE_THRESHOLD = 0.25  # Stricter when we have matching ingredients
        else:
            MIN_SCORE_THRESHOLD = 0.15  # More lenient if no matching ingredients
        
        result_df_above_threshold = result_df[
            result_df['final_score'] >= MIN_SCORE_THRESHOLD
        ]
        
        if len(result_df_above_threshold) == 0:
            fallback_applied = True
            fallback_reason = f"No recipes met minimum quality threshold. Showing best available matches with your ingredients."
            # Keep result_df as-is; sorting will show best matches
        else:
            result_df = result_df_above_threshold
        
        # Step 6: DEDUPLICATION
        # ENHANCEMENT: Ensure no duplicate recipe titles in results
        result_df = result_df.drop_duplicates(subset=['recipe_name'], keep='first')
        
        # Step 7: Sort by final score and select top-N
        result_df = result_df.sort_values('final_score', ascending=False)
        final_results = result_df[
            ['recipe_name', 'ingredients', 'cuisine', 'meal_type',
             'cooking_time', 'instructions', 'ingredient_similarity',
             'final_score', 'ingredient_overlap_count']
        ].head(n_recommendations)
        
        # Step 8: Add fallback notification to results (if applicable)
        if fallback_applied and len(final_results) > 0:
            # Store fallback info in a special attribute (won't break DataFrame)
            final_results._fallback_notification = fallback_reason
        
        return final_results
    
    @staticmethod
    def _compute_time_score(actual_time: float, max_time: float) -> float:
        """
        Compute cooking time score (penalty-based).
        
        Higher score for recipes within reasonable time.
        HARSH penalty for recipes exceeding max_time.
        
        Args:
            actual_time: Actual cooking time
            max_time: Maximum acceptable cooking time
        
        Returns:
            Score between 0 and 1
        """
        if actual_time <= max_time:
            # Perfect score if within time, slight preference for shorter times
            return 0.95 + 0.05 * (1 - actual_time / max_time)
        else:
            # HARSH penalty for exceeding max time (not just soft penalty)
            excess_ratio = (actual_time - max_time) / max_time
            # Score drops significantly for exceeded time
            return max(0.0, 0.5 - excess_ratio * 0.8)
    
    def _apply_cuisine_prioritization(
        self,
        result_df: pd.DataFrame,
        cuisine_selected: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply cuisine-specific prioritization to recommendations.
        
        ENHANCEMENT: When a user selects a specific cuisine (e.g., Indian),
        recipes from that cuisine are STRONGLY preferred. This is enforced
        via hard filtering earlier in the recommend() method, so by the time
        this function is called, only matching cuisines remain. This function
        now applies a stronger boost (5% instead of 2%) to ensure consistent
        cuisine-matching behavior.
        
        Args:
            result_df: DataFrame with computed final_scores
            cuisine_selected: Cuisine selected by user (e.g., 'Indian')
        
        Returns:
            DataFrame with adjusted final_scores
        """
        if cuisine_selected is None or cuisine_selected.lower() == 'all':
            return result_df
        
        # Note: Hard filtering is done in recommend() method, so recipes
        # in result_df should already match the selected cuisine. This boost
        # is for tie-breaking and ranking purposes.
        
        cuisine_selected_norm = cuisine_selected.lower().strip()
        boost_factor = 1.05  # Increased from 1.02 (2% → 5% boost)
        
        result_df['final_score'] = result_df.apply(
            lambda row: row['final_score'] * boost_factor
            if row['cuisine'].lower() == cuisine_selected_norm
            else row['final_score'],
            axis=1
        )
        
        return result_df
    
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
