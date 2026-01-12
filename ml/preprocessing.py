"""
Preprocessing module for recipe data.

Handles:
- Real dataset loading from Kaggle (64k recipes)
- JSON parsing of ingredients and directions
- Text cleaning (lowercase, punctuation removal)
- Ingredient tokenization and normalization
- Feature preparation for ML pipeline

NOTE: The earlier synthetic data generation was replaced with a
real-world recipe dataset imported programmatically using the
Kaggle API to improve realism, reproducibility, and internship-grade
data integrity.
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path


class RecipePreprocessor:
    """Preprocesses recipe data from Kaggle dataset for ML pipeline."""
    
    # Required columns from the Kaggle dataset
    REQUIRED_COLUMNS = {
        'recipe_title', 'category', 'subcategory', 'description',
        'ingredients', 'directions', 'num_ingredients', 'num_steps'
    }
    
    VALID_CUISINES = {
        'indian', 'chinese', 'italian', 'mexican', 'thai',
        'french', 'japanese', 'american', 'mediterranean',
        'spanish', 'korean', 'vietnamese', 'greek', 'turkish',
        'asian', 'middle eastern', 'african', 'european'
    }
    
    VALID_MEAL_TYPES = {
        'breakfast', 'lunch', 'dinner', 'snack', 'appetizer', 'dessert',
        'main course', 'side dish', 'soup', 'salad'
    }
    
    def __init__(self):
        """Initialize preprocessor."""
        self.df = None
        self.original_df = None
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess real recipe dataset from Kaggle.
        
        This method:
        1. Loads the 64k recipe dataset from data/1_Recipe_csv.csv
        2. Validates required columns
        3. Parses JSON ingredients and directions
        4. Cleans and prepares text features
        5. Derives cuisine and meal type from category/subcategory
        6. Produces output compatible with ML pipeline
        
        Args:
            csv_path: Path to the Kaggle recipe CSV file
                      (should be data/1_Recipe_csv.csv)
        
        Returns:
            Preprocessed DataFrame with columns required by Phase 3:
            - recipe_title, category, subcategory
            - ingredients_text (cleaned ingredient string for TF-IDF)
            - ingredients_list (parsed JSON list)
            - directions_list (parsed JSON list)
            - cuisine, meal_type
            - num_ingredients, num_steps
            - description
        
        Raises:
            FileNotFoundError: If CSV file not found
            KeyError: If required columns missing
            ValueError: If JSON parsing fails
        """
        # Step 2.1: Load CSV with validation
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. "
                f"Please ensure data/1_Recipe_csv.csv is available."
            )
        
        self.original_df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Validate required columns
        missing_cols = self.REQUIRED_COLUMNS - set(self.original_df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {set(self.original_df.columns)}"
            )
        
        # Create working copy
        self.df = self.original_df.copy()
        
        # Step 2.2: Parse JSON fields (ingredients and directions)
        self.df['ingredients_list'] = self.df['ingredients'].apply(
            self._parse_json_list
        )
        self.df['directions_list'] = self.df['directions'].apply(
            self._parse_json_list
        )
        
        # Step 2.4: Prepare text feature for ML (ingredients into cleaned string)
        self.df['ingredients_text'] = self.df['ingredients_list'].apply(
            self._prepare_ingredients_text
        )
        
        # Step 2.4: Clean and normalize ingredients_text (required by Phase 3)
        self.df['ingredients_cleaned'] = self.df['ingredients_text'].apply(
            self._clean_ingredients
        )
        
        # Derive cuisine from category and subcategory
        self.df['cuisine'] = self.df.apply(
            self._extract_cuisine, axis=1
        )
        
        # Derive meal_type from category and subcategory
        self.df['meal_type'] = self.df.apply(
            self._extract_meal_type, axis=1
        )
        
        # Create default cooking_time (not in Kaggle dataset)
        # Use heuristic: num_steps * 5 minutes as estimate
        self.df['cooking_time'] = (
            self.df['num_steps'].fillna(5) * 5
        ).clip(5, 180)  # Clamp to 5-180 minutes
        
        # Create aliases for backward compatibility with Phase 3
        # Kaggle dataset uses 'recipe_title' and 'directions'
        # but downstream expects 'recipe_name' and 'instructions'
        self.df['recipe_name'] = self.df['recipe_title']
        self.df['instructions'] = self.df['directions']
        
        # Also rename ingredients to match expected format
        self.df['ingredients'] = self.df['ingredients_text']
        
        # Drop rows with empty ingredients (critical for ML)
        self.df = self.df[self.df['ingredients_cleaned'].str.len() > 0].reset_index(drop=True)
        
        return self.df
    
    def _parse_json_list(self, json_str) -> list:
        """
        Parse stringified JSON list safely.
        
        Args:
            json_str: Stringified JSON list or None
        
        Returns:
            Parsed Python list, or empty list if parsing fails
        """
        if not isinstance(json_str, str) or not json_str.strip():
            return []
        
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _prepare_ingredients_text(self, ingredients_list: list) -> str:
        """
        Convert ingredient list to single clean string.
        
        Args:
            ingredients_list: Parsed list of ingredients
        
        Returns:
            Space-separated ingredient string
        """
        if not isinstance(ingredients_list, list) or len(ingredients_list) == 0:
            return ""
        
        # Join all ingredients with space
        return ' '.join(str(ing).strip() for ing in ingredients_list if ing)
    
    def _clean_ingredients(self, ingredients_str: str) -> str:
        """
        Clean ingredient text for TF-IDF vectorization.
        
        Applies:
        - Lowercase
        - Punctuation removal
        - Whitespace normalization
        
        Args:
            ingredients_str: Raw ingredient string
        
        Returns:
            Cleaned ingredient string (ready for TF-IDF)
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
    
    def _extract_cuisine(self, row) -> str:
        """
        Extract cuisine from category/subcategory fields.
        
        Args:
            row: DataFrame row with 'category' and 'subcategory'
        
        Returns:
            Normalized cuisine string
        """
        category = str(row.get('category', '')).lower().strip()
        subcategory = str(row.get('subcategory', '')).lower().strip()
        
        # Check category first
        if category in self.VALID_CUISINES:
            return category
        
        # Check subcategory
        if subcategory in self.VALID_CUISINES:
            return subcategory
        
        # Try to find partial match
        combined = f"{category} {subcategory}"
        for valid_cuisine in self.VALID_CUISINES:
            if valid_cuisine in combined:
                return valid_cuisine
        
        # Try category substring match
        if category:
            for valid_cuisine in self.VALID_CUISINES:
                if valid_cuisine in category or category in valid_cuisine:
                    return valid_cuisine
        
        return 'other'
    
    def _extract_meal_type(self, row) -> str:
        """
        Extract meal type from category/subcategory fields.
        
        Args:
            row: DataFrame row with 'category' and 'subcategory'
        
        Returns:
            Normalized meal type string
        """
        category = str(row.get('category', '')).lower().strip()
        subcategory = str(row.get('subcategory', '')).lower().strip()
        
        # Check category first
        if category in self.VALID_MEAL_TYPES:
            return category
        
        # Check subcategory
        if subcategory in self.VALID_MEAL_TYPES:
            return subcategory
        
        # Try to find partial match
        combined = f"{category} {subcategory}"
        for valid_type in self.VALID_MEAL_TYPES:
            if valid_type in combined:
                return valid_type
        
        # Try category substring match
        if category:
            for valid_type in self.VALID_MEAL_TYPES:
                if valid_type in category or category in valid_type:
                    return valid_type
        
        return 'other'
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics for real Kaggle dataset.
        
        Returns:
            Dictionary with dataset stats
        """
        if self.df is None or len(self.df) == 0:
            return {}
        
        return {
            'total_recipes': len(self.df),
            'unique_cuisines': self.df['cuisine'].nunique(),
            'unique_meal_types': self.df['meal_type'].nunique(),
            'avg_cooking_time': float(self.df['cooking_time'].mean()),
            'avg_ingredients': float(self.df['num_ingredients'].mean()),
            'avg_steps': float(self.df['num_steps'].mean()),
            'cuisines': self.df['cuisine'].unique().tolist(),
            'meal_types': self.df['meal_type'].unique().tolist()
        }
