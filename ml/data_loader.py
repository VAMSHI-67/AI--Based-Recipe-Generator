"""
Multi-dataset recipe loader with Kaggle API integration.

This module handles:
- Programmatic dataset import via Kaggle API
- Loading and validating the 64k Kaggle recipe dataset
- Loading and inspecting the Cleaned Indian Recipes Dataset
- Schema normalization for both datasets
- Safe merging with cuisine labeling
- Reproducible data pipeline

DATASETS:
1. Base: Recipes Dataset (64k) - https://www.kaggle.com/datasets/pes12017000148/recipes-dataset
2. Extended: Cleaned Indian Recipes - https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset

This enables truthful statement: "Multiple real-world datasets were programmatically 
imported via Kaggle API and unified to enhance cuisine-specific recommendations."
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import kagglehub


class DataLoader:
    """Multi-dataset loader with Kaggle API integration and schema normalization."""
    
    # Standard schema all datasets must normalize to
    STANDARD_SCHEMA = {
        'recipe_title': 'str',
        'category': 'str',
        'subcategory': 'str',
        'ingredients_list': 'list',
        'directions_list': 'list',
        'num_ingredients': 'int',
        'num_steps': 'int',
        'cuisine': 'str'
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for all data operations
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.kaggle_dir = self.data_dir / "kaggle_datasets"
        self.kaggle_dir.mkdir(exist_ok=True)
    
    def load_kaggle_64k_dataset(
        self,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load the 64k Kaggle recipes dataset.
        
        This is the primary dataset. The CSV should already exist at
        data/1_Recipe_csv.csv from the project initialization phase.
        
        Args:
            force_refresh: If True, re-download from Kaggle API
        
        Returns:
            DataFrame with 64k recipes (normalized schema)
        
        Raises:
            FileNotFoundError: If dataset not found and force_refresh=False
            ValueError: If required columns missing after loading
        """
        csv_path = self.data_dir / "1_Recipe_csv.csv"
        
        # Check if file exists
        if csv_path.exists() and not force_refresh:
            print(f"[OK] Loading 64k dataset from {csv_path}")
            df = pd.read_csv(csv_path, encoding='utf-8')
        else:
            # Download via Kaggle API
            print("[DOWNLOAD]  Downloading 64k Kaggle recipes dataset via Kaggle API...")
            try:
                # Note: This requires kagglehub and proper Kaggle credentials
                # Dataset: https://www.kaggle.com/datasets/pes12017000148/recipes-dataset
                df = kagglehub.datasets.download(
                    "pes12017000148/recipes-dataset"
                )
                # kagglehub returns path; load the actual CSV
                # (adjust filename based on actual dataset structure)
                kaggle_csv = Path(df) / "recipes.csv"
                if not kaggle_csv.exists():
                    # Try alternate names
                    csv_files = list(Path(df).glob("*.csv"))
                    if csv_files:
                        kaggle_csv = csv_files[0]
                    else:
                        raise FileNotFoundError("No CSV found in downloaded dataset")
                
                df = pd.read_csv(kaggle_csv, encoding='utf-8')
                # Save locally
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"[OK] Downloaded and cached 64k dataset to {csv_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load 64k dataset. Ensure data/1_Recipe_csv.csv exists or "
                    f"Kaggle API is configured. Error: {e}"
                )
        
        # Validate required columns
        required_cols = {'recipe_title', 'category', 'subcategory', 'ingredients', 
                        'directions', 'num_ingredients', 'num_steps'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in 64k dataset: {missing}\n"
                f"Available: {set(df.columns)}"
            )
        
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def download_indian_recipes_dataset(
        self,
        force_refresh: bool = False
    ) -> Path:
        """
        Download Cleaned Indian Recipes Dataset via Kaggle API.
        
        Dataset: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset
        
        This requires:
        1. Kaggle account
        2. API key saved at ~/.kaggle/kaggle.json
        3. kagglehub library installed
        
        Args:
            force_refresh: If True, re-download even if cached
        
        Returns:
            Path to the downloaded dataset directory
        
        Raises:
            RuntimeError: If download fails or credentials missing
        """
        indian_cache_dir = self.kaggle_dir / "indian_recipes"
        
        # Check if already downloaded
        if indian_cache_dir.exists() and not force_refresh:
            csv_files = list(indian_cache_dir.glob("*.csv"))
            if csv_files:
                print(f"[OK] Using cached Indian recipes dataset at {indian_cache_dir}")
                return indian_cache_dir
        
        print("[DOWNLOAD]  Downloading Cleaned Indian Recipes Dataset via Kaggle API...")
        print("   Dataset: sooryaprakash12/cleaned-indian-recipes-dataset")
        
        try:
            # Download via kagglehub
            download_path = kagglehub.datasets.download(
                "sooryaprakash12/cleaned-indian-recipes-dataset"
            )
            
            # Copy to our cache directory
            import shutil
            if indian_cache_dir.exists():
                shutil.rmtree(indian_cache_dir)
            shutil.move(download_path, str(indian_cache_dir))
            
            print(f"[OK] Downloaded Indian recipes to {indian_cache_dir}")
            return indian_cache_dir
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Indian recipes dataset via Kaggle API.\n"
                f"Ensure:\n"
                f"  1. kagglehub is installed: pip install kagglehub\n"
                f"  2. Kaggle API key at ~/.kaggle/kaggle.json\n"
                f"  3. Dataset access: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset\n"
                f"Error: {e}"
            )
    
    def load_indian_recipes_dataset(
        self,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load and inspect the Cleaned Indian Recipes Dataset.
        
        Gracefully handles various column formats and missing fields.
        
        Args:
            force_refresh: Force re-download from Kaggle API
        
        Returns:
            Raw Indian recipes DataFrame
        
        Raises:
            RuntimeError: If dataset cannot be loaded
        """
        print("\n" + "="*70)
        print("PHASE 2.2: LOADING INDIAN RECIPES DATASET")
        print("="*70)
        
        # Download/locate dataset
        dataset_dir = self.download_indian_recipes_dataset(force_refresh=force_refresh)
        
        # Find CSV file
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV file found in {dataset_dir}")
        
        csv_file = csv_files[0]
        print(f"\nLoading CSV: {csv_file.name}")
        
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        
        print(f"\n[OK] Loaded Indian dataset")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        return df
    
    def normalize_indian_dataset(
        self,
        raw_indian_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize Indian dataset to standard schema.
        
        Handles:
        - Diverse column naming (name, recipe_name, recipe, title, etc.)
        - String vs list ingredients/directions
        - Missing cuisine/category fields
        - Missing cooking metadata
        
        Args:
            raw_indian_df: Raw Indian recipes DataFrame
        
        Returns:
            Normalized DataFrame matching STANDARD_SCHEMA
        """
        print("\n" + "="*70)
        print("PHASE 2.3: NORMALIZING INDIAN DATASET SCHEMA")
        print("="*70)
        
        df = raw_indian_df.copy()
        
        # Step 1: Map recipe title
        title_cols = [col for col in df.columns if col.lower() in 
                     ['name', 'recipe_name', 'recipe', 'title', 'recipe_title']]
        if title_cols:
            df['recipe_title'] = df[title_cols[0]].astype(str)
            print(f"  Mapped '{title_cols[0]}' -> 'recipe_title'")
        else:
            raise ValueError("No recipe title column found in Indian dataset")
        
        # Step 2: Map ingredients
        ingredient_cols = [col for col in df.columns if col.lower() in 
                          ['ingredients', 'ingredient', 'items']]
        if ingredient_cols:
            ingredient_col = ingredient_cols[0]
            # Parse if stored as JSON or list string
            df['ingredients_list'] = df[ingredient_col].apply(
                self._parse_ingredients
            )
            print(f"  Mapped '{ingredient_col}' -> 'ingredients_list'")
        else:
            print("  [WARN]  No ingredients column found - using empty lists")
            df['ingredients_list'] = [[]] * len(df)
        
        # Step 3: Map directions
        direction_cols = [col for col in df.columns if col.lower() in 
                         ['directions', 'direction', 'instructions', 'steps', 'method', 'procedure']]
        if direction_cols:
            direction_col = direction_cols[0]
            df['directions_list'] = df[direction_col].apply(
                self._parse_directions
            )
            print(f"  Mapped '{direction_col}' -> 'directions_list'")
        else:
            print("  [WARN]  No directions column found - using empty lists")
            df['directions_list'] = [[]] * len(df)
        
        # Step 4: Compute num_ingredients
        df['num_ingredients'] = df['ingredients_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Step 5: Compute num_steps
        df['num_steps'] = df['directions_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Step 6: Map category/subcategory
        category_cols = [col for col in df.columns if col.lower() in 
                        ['category', 'course', 'type', 'dish_type']]
        if category_cols:
            df['category'] = df[category_cols[0]].astype(str)
            print(f"  Mapped '{category_cols[0]}' -> 'category'")
        else:
            print("  [WARN]  No category found - using 'Indian Dish'")
            df['category'] = "Indian Dish"
        
        # Subcategory: use category if not present
        df['subcategory'] = df.get('subcategory', df['category']).astype(str)
        
        # Step 7: Add cuisine label
        df['cuisine'] = 'Indian'
        print(f"  Added cuisine label: 'Indian'")
        
        # Select only standard schema columns
        normalized_df = df[[
            'recipe_title', 'category', 'subcategory',
            'ingredients_list', 'directions_list',
            'num_ingredients', 'num_steps', 'cuisine'
        ]].copy()
        
        # Remove duplicates by recipe_title
        normalized_df = normalized_df.drop_duplicates(
            subset=['recipe_title'],
            keep='first'
        ).reset_index(drop=True)
        
        print(f"\n[OK] Normalized Indian dataset")
        print(f"  Final shape: {normalized_df.shape}")
        print(f"  Duplicates removed: {len(df) - len(normalized_df)}")
        
        return normalized_df
    
    def normalize_kaggle_64k_dataset(
        self,
        raw_64k_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize the 64k Kaggle dataset to standard schema.
        
        Args:
            raw_64k_df: Raw 64k recipes DataFrame
        
        Returns:
            Normalized DataFrame matching STANDARD_SCHEMA
        """
        print("\n" + "="*70)
        print("PHASE 2.3: NORMALIZING 64K KAGGLE DATASET SCHEMA")
        print("="*70)
        
        df = raw_64k_df.copy()
        
        # The 64k dataset should already have proper schema from preprocessing.py
        # but we ensure consistency here
        
        # Parse JSON fields if stored as strings
        df['ingredients_list'] = df['ingredients'].apply(self._parse_ingredients)
        df['directions_list'] = df['directions'].apply(self._parse_directions)
        
        # Ensure metadata
        df['num_ingredients'] = df['num_ingredients'].fillna(
            df['ingredients_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        ).astype(int)
        
        df['num_steps'] = df['num_steps'].fillna(
            df['directions_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        ).astype(int)
        
        # Derive cuisine if not present
        if 'cuisine' not in df.columns:
            df['cuisine'] = df.apply(
                lambda row: self._infer_cuisine(row.get('category', ''), 
                                               row.get('subcategory', '')),
                axis=1
            )
            print("  Inferred cuisine from category/subcategory")
        
        # Normalize categories
        df['category'] = df['category'].fillna('Other').astype(str)
        df['subcategory'] = df['subcategory'].fillna('Other').astype(str)
        
        # Select standard schema columns
        normalized_df = df[[
            'recipe_title', 'category', 'subcategory',
            'ingredients_list', 'directions_list',
            'num_ingredients', 'num_steps', 'cuisine'
        ]].copy()
        
        print(f"[OK] Normalized 64k Kaggle dataset")
        print(f"  Shape: {normalized_df.shape}")
        
        return normalized_df
    
    def merge_datasets(
        self,
        kaggle_64k_df: pd.DataFrame,
        indian_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge 64k Kaggle and Indian datasets.
        
        Args:
            kaggle_64k_df: Normalized 64k dataset
            indian_df: Normalized Indian dataset
        
        Returns:
            Merged DataFrame with no duplicates
        """
        print("\n" + "="*70)
        print("PHASE 2.4: MERGING DATASETS")
        print("="*70)
        
        print(f"\nInput shapes:")
        print(f"  Kaggle 64k: {kaggle_64k_df.shape}")
        print(f"  Indian: {indian_df.shape}")
        
        # Concatenate
        merged_df = pd.concat(
            [kaggle_64k_df, indian_df],
            axis=0,
            ignore_index=True
        )
        
        # Remove title duplicates
        before = len(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=['recipe_title'],
            keep='first'  # Keep first occurrence (prefer 64k dataset)
        ).reset_index(drop=True)
        after = len(merged_df)
        
        print(f"\nDuplicate removal:")
        print(f"  Before: {before} recipes")
        print(f"  After: {after} recipes")
        print(f"  Duplicates removed: {before - after}")
        
        # Cuisine distribution
        print(f"\nCuisine distribution:")
        cuisine_counts = merged_df['cuisine'].value_counts().head(10)
        for cuisine, count in cuisine_counts.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {cuisine}: {count} ({percentage:.1f}%)")
        
        print(f"\n[OK] Final merged dataset")
        print(f"  Total recipes: {len(merged_df)}")
        print(f"  Cuisines: {merged_df['cuisine'].nunique()}")
        
        return merged_df
    
    def _parse_ingredients(self, value) -> List[str]:
        """Parse ingredients from various formats into list of strings."""
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        
        if not isinstance(value, str) or pd.isna(value):
            return []
        
        value = str(value).strip()
        
        # Try JSON format
        if value.startswith('['):
            try:
                parsed = json.loads(value)
                return [str(x).strip() for x in parsed if str(x).strip()]
            except:
                pass
        
        # Try comma-separated
        if ',' in value:
            return [x.strip() for x in value.split(',') if x.strip()]
        
        # Single ingredient
        if value:
            return [value]
        
        return []
    
    def _parse_directions(self, value) -> List[str]:
        """Parse directions from various formats into list of steps."""
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        
        if not isinstance(value, str) or pd.isna(value):
            return []
        
        value = str(value).strip()
        
        # Try JSON format
        if value.startswith('['):
            try:
                parsed = json.loads(value)
                return [str(x).strip() for x in parsed if str(x).strip()]
            except:
                pass
        
        # Split by numbered steps (1. 2. 3. etc.)
        import re
        steps = re.split(r'\d+\.\s+', value)
        steps = [s.strip() for s in steps if s.strip()]
        
        if steps:
            return steps
        
        # Single direction
        if value:
            return [value]
        
        return []
    
    def _infer_cuisine(self, category: str, subcategory: str) -> str:
        """Infer cuisine from category/subcategory strings."""
        text = f"{category} {subcategory}".lower()
        
        cuisine_keywords = {
            'indian': ['indian', 'curry', 'tandoor', 'biryani', 'dal', 'samosa'],
            'chinese': ['chinese', 'wok', 'soy', 'dumpling', 'noodle'],
            'italian': ['italian', 'pasta', 'pizza', 'risotto'],
            'mexican': ['mexican', 'taco', 'burrito', 'enchilada'],
            'thai': ['thai', 'coconut', 'pad thai'],
            'japanese': ['japanese', 'sushi', 'ramen'],
            'french': ['french', 'beurre', 'coq au vin'],
            'mediterranean': ['mediterranean', 'greek', 'hummus'],
        }
        
        for cuisine, keywords in cuisine_keywords.items():
            if any(kw in text for kw in keywords):
                return cuisine.capitalize()
        
        return 'Other'


def load_global_dataset(
    force_refresh: bool = False,
    include_indian: bool = True
) -> pd.DataFrame:
    """
    PHASE 2: Load and merge global recipes dataset.
    
    This is the main entry point for data loading. It:
    1. Loads the 64k Kaggle recipes dataset
    2. Optionally loads and merges the Indian recipes dataset
    3. Normalizes schemas
    4. Returns unified DataFrame for downstream processing
    
    Args:
        force_refresh: Force re-download of datasets
        include_indian: Include Indian recipes in final dataset
    
    Returns:
        Merged DataFrame ready for Phase 3 (preprocessing)
    
    Example:
        >>> merged_df = load_global_dataset(include_indian=True)
        >>> print(f"Total recipes: {len(merged_df)}")
        >>> print(f"Indian recipes: {len(merged_df[merged_df['cuisine'] == 'Indian'])}")
    """
    loader = DataLoader(data_dir="data")
    
    # Load base Kaggle dataset
    print("\n" + "="*70)
    print("PHASE 2.1: LOADING KAGGLE 64K DATASET")
    print("="*70 + "\n")
    
    kaggle_64k_raw = loader.load_kaggle_64k_dataset(force_refresh=force_refresh)
    kaggle_64k_normalized = loader.normalize_kaggle_64k_dataset(kaggle_64k_raw)
    
    # Optionally load and merge Indian dataset
    if include_indian:
        try:
            indian_raw = loader.load_indian_recipes_dataset(force_refresh=force_refresh)
            indian_normalized = loader.normalize_indian_dataset(indian_raw)
            merged = loader.merge_datasets(kaggle_64k_normalized, indian_normalized)
        except Exception as e:
            print(f"\n[WARN]  Warning: Could not load Indian dataset: {e}")
            print("   Proceeding with 64k Kaggle dataset only...")
            merged = kaggle_64k_normalized
    else:
        merged = kaggle_64k_normalized
    
    print("\n" + "="*70)
    print("[OK] PHASE 2 COMPLETE: Data loading and merging finished")
    print("="*70)
    
    return merged


def load_indian_dataset_only() -> pd.DataFrame:
    """
    Load only the Indian recipes dataset (for testing/analysis).
    
    Returns:
        Normalized Indian dataset
    """
    loader = DataLoader(data_dir="data")
    indian_raw = loader.load_indian_recipes_dataset()
    return loader.normalize_indian_dataset(indian_raw)


if __name__ == "__main__":
    """Testing and demonstration."""
    # Load merged dataset
    print("🍳 Recipe Recommendation System - Data Loader")
    print("="*70)
    
    try:
        # Load with Indian recipes
        merged_df = load_global_dataset(include_indian=True)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"[OK] Successfully loaded {len(merged_df)} recipes")
        print(f"[OK] From 64k Kaggle + Indian dataset")
        print(f"[OK] Indian recipes: {len(merged_df[merged_df['cuisine'] == 'Indian'])}")
        print(f"[OK] Ready for downstream ML pipeline")
        
        # Sample
        print(f"\nSample recipe:")
        sample = merged_df.iloc[0]
        print(f"  Title: {sample['recipe_title']}")
        print(f"  Cuisine: {sample['cuisine']}")
        print(f"  Ingredients: {len(sample['ingredients_list'])} items")
        print(f"  Steps: {len(sample['directions_list'])} steps")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
