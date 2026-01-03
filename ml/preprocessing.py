"""
Preprocessing module for recipe data.

Handles:
- Data loading from CSV
- Text cleaning (lowercase, punctuation removal)
- Ingredient tokenization and normalization
- Categorical field standardization
- Missing value handling
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


class RecipePreprocessor:
    """Preprocesses recipe data for ML pipeline."""
    
    VALID_CUISINES = {
        'indian', 'chinese', 'italian', 'mexican', 'thai',
        'french', 'japanese', 'american', 'mediterranean',
        'spanish', 'korean', 'vietnamese', 'greek', 'turkish'
    }
    
    VALID_MEAL_TYPES = {
        'breakfast', 'lunch', 'dinner', 'snack', 'appetizer', 'dessert'
    }
    
    def __init__(self):
        """Initialize preprocessor."""
        self.df = None
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess recipe dataset.
        
        Args:
            csv_path: Path to recipes.csv file
        
        Returns:
            Preprocessed DataFrame
        """
        # Load data
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Handle missing values
        self._handle_missing_values()
        
        # Clean ingredients
        self.df['ingredients_cleaned'] = self.df['ingredients'].apply(
            self._clean_ingredients
        )
        
        # Normalize categorical fields
        self.df['cuisine'] = self.df['cuisine'].apply(
            self._normalize_cuisine
        )
        self.df['meal_type'] = self.df['meal_type'].apply(
            self._normalize_meal_type
        )
        
        # Ensure cooking_time is numeric
        self.df['cooking_time'] = pd.to_numeric(
            self.df['cooking_time'],
            errors='coerce'
        ).fillna(30)
        
        return self.df
    
    def _handle_missing_values(self) -> None:
        """Handle missing values in dataset."""
        # Fill missing ingredients
        if 'ingredients' in self.df.columns:
            self.df['ingredients'] = self.df['ingredients'].fillna('')
        
        # Fill missing cuisine
        if 'cuisine' in self.df.columns:
            self.df['cuisine'] = self.df['cuisine'].fillna('other')
        
        # Fill missing meal_type
        if 'meal_type' in self.df.columns:
            self.df['meal_type'] = self.df['meal_type'].fillna('other')
        
        # Fill missing cooking_time
        if 'cooking_time' in self.df.columns:
            self.df['cooking_time'] = self.df['cooking_time'].fillna(30)
        
        # Fill missing instructions
        if 'instructions' in self.df.columns:
            self.df['instructions'] = self.df['instructions'].fillna('No instructions provided')
    
    def _clean_ingredients(self, ingredients_str: str) -> str:
        """
        Clean ingredient text.
        
        Args:
            ingredients_str: Raw ingredient string
        
        Returns:
            Cleaned ingredient string
        """
        if not isinstance(ingredients_str, str) or not ingredients_str.strip():
            return ""
        
        # Convert to lowercase
        text = ingredients_str.lower()
        
        # Remove special characters but keep spaces and commas
        text = re.sub(r'[^a-z0-9\s,]', '', text)
        
        # Tokenize by comma or space
        tokens = re.split(r'[,\s]+', text)
        
        # Remove empty tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        
        # Join back with space
        return ' '.join(tokens)
    
    def _normalize_cuisine(self, cuisine_str: str) -> str:
        """
        Normalize cuisine field.
        
        Args:
            cuisine_str: Raw cuisine string
        
        Returns:
            Normalized cuisine
        """
        if not isinstance(cuisine_str, str):
            return 'other'
        
        cuisine = cuisine_str.lower().strip()
        
        # Check if valid cuisine
        if cuisine in self.VALID_CUISINES:
            return cuisine
        
        # Try to find partial match
        for valid_cuisine in self.VALID_CUISINES:
            if valid_cuisine in cuisine or cuisine in valid_cuisine:
                return valid_cuisine
        
        return 'other'
    
    def _normalize_meal_type(self, meal_type_str: str) -> str:
        """
        Normalize meal type field.
        
        Args:
            meal_type_str: Raw meal type string
        
        Returns:
            Normalized meal type
        """
        if not isinstance(meal_type_str, str):
            return 'other'
        
        meal_type = meal_type_str.lower().strip()
        
        # Check if valid meal type
        if meal_type in self.VALID_MEAL_TYPES:
            return meal_type
        
        # Try to find partial match
        for valid_type in self.VALID_MEAL_TYPES:
            if valid_type in meal_type or meal_type in valid_type:
                return valid_type
        
        return 'other'
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset stats
        """
        if self.df is None:
            return {}
        
        return {
            'total_recipes': len(self.df),
            'unique_cuisines': self.df['cuisine'].nunique(),
            'unique_meal_types': self.df['meal_type'].nunique(),
            'avg_cooking_time': self.df['cooking_time'].mean(),
            'cuisines': self.df['cuisine'].unique().tolist(),
            'meal_types': self.df['meal_type'].unique().tolist()
        }
