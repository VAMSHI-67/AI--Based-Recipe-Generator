"""
AI-Based Recipe Recommendation System
Main Streamlit application entrypoint.

This module loads recipe datasets and provides an interactive interface
for users to get recommendations based on their preferences.

PHASE 2: Data Loading
- Loads 64k Kaggle recipes dataset
- Optionally loads Cleaned Indian Recipes dataset via Kaggle API
- Normalizes schemas and merges into unified DataFrame

PHASE 3+: Preprocessing, ML, and UI
- Uses TF-IDF + cosine similarity for ingredient-based recommendations
- Applies preference scoring and cuisine-specific prioritization
- Provides Streamlit interface with video recommendations
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml.data_loader import load_global_dataset
from ml.preprocessing import RecipePreprocessor
from ml.recommender import RecipeRecommender
from ml.scoring import PreferenceScorer
from utils.youtube import YouTubeRecommendationGenerator

# Page configuration
st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "recommender" not in st.session_state:
    st.session_state.recommender = None
    st.session_state.preprocessor = None
    st.session_state.scorer = None
    st.session_state.youtube_gen = None
    st.session_state.recipes_df = None

def load_system():
    """
    Load and initialize the recommendation system.
    
    PHASE 2: Data Loading & Merging
    - Loads 64k Kaggle recipes
    - Loads Cleaned Indian Recipes via Kaggle API
    - Merges and normalizes to unified schema
    
    PHASE 3+: Preprocessing and Model Building
    - Applies preprocessing (JSON parsing, cleaning, feature engineering)
    - Builds TF-IDF model for content-based recommendations
    """
    try:
        st.write("**PHASE 2: Loading and merging recipe datasets...**")
        
        # Phase 2: Load merged dataset from multiple sources
        merged_df = load_global_dataset(
            force_refresh=False,
            include_indian=True  # Include Indian recipes
        )
        
        st.write(f"✓ Loaded {len(merged_df)} recipes from merged dataset")
        st.write(f"  - Indian recipes: {len(merged_df[merged_df['cuisine'] == 'Indian'])}")
        
        # Phase 3+: Preprocess the merged dataset
        st.write("**PHASE 3: Preprocessing and feature engineering...**")
        
        # Create a temporary CSV from merged data for preprocessing
        # (PreprocessorRecipe expects CSV, so we work with the existing flow)
        st.session_state.preprocessor = RecipePreprocessor()
        
        # Instead of loading from CSV, we'll directly assign and preprocess
        st.session_state.preprocessor.df = merged_df.copy()
        st.session_state.preprocessor.original_df = merged_df.copy()
        
        # Parse and prepare features (mimicking load_and_preprocess)
        preprocessor = st.session_state.preprocessor
        
        # Create aliases for downstream compatibility
        if 'ingredients_list' in merged_df.columns:
            preprocessor.df['ingredients_list'] = merged_df['ingredients_list']
        
        if 'directions_list' in merged_df.columns:
            preprocessor.df['directions_list'] = merged_df['directions_list']
        
        # Prepare ingredients text for TF-IDF
        preprocessor.df['ingredients_text'] = preprocessor.df['ingredients_list'].apply(
            preprocessor._prepare_ingredients_text
        )
        preprocessor.df['ingredients_cleaned'] = preprocessor.df['ingredients_text'].apply(
            preprocessor._clean_ingredients
        )
        
        # Create aliases for downstream compatibility
        preprocessor.df['recipe_name'] = preprocessor.df['recipe_title']
        preprocessor.df['instructions'] = preprocessor.df['directions_list'].apply(
            lambda x: '\n'.join(x) if isinstance(x, list) else str(x)
        )
        preprocessor.df['ingredients'] = preprocessor.df['ingredients_text']
        
        # Add cooking_time if missing
        if 'cooking_time' not in preprocessor.df.columns:
            preprocessor.df['cooking_time'] = (
                preprocessor.df['num_steps'].fillna(5) * 5
            ).clip(5, 180)
        
        # Add meal_type if missing
        if 'meal_type' not in preprocessor.df.columns:
            preprocessor.df['meal_type'] = preprocessor.df['category'].fillna('other')
        
        # Remove empty recipes
        recipes_df = preprocessor.df[
            preprocessor.df['ingredients_cleaned'].str.len() > 0
        ].reset_index(drop=True)
        
        st.write(f"✓ Preprocessed dataset: {len(recipes_df)} recipes")
        
        # Phase 4: Build ML model
        st.write("**PHASE 4: Building TF-IDF and recommendation model...**")
        
        st.session_state.preprocessor = preprocessor
        st.session_state.recipes_df = recipes_df
        
        # Initialize recommender
        st.session_state.recommender = RecipeRecommender(recipes_df)
        st.session_state.recommender.build_tfidf_model()
        st.write("✓ Built TF-IDF vectorizer")
        
        # Initialize scorer
        st.session_state.scorer = PreferenceScorer()
        st.write("✓ Initialized preference scorer")
        
        # Initialize YouTube generator
        st.session_state.youtube_gen = YouTubeRecommendationGenerator()
        st.write("✓ Initialized YouTube recommendations")
        
        st.success("✓ System ready! All phases complete.")
        
        return recipes_df
        
    except Exception as e:
        st.error(f"Error loading system: {e}")
        import traceback
        st.write(traceback.format_exc())
        return None

def main():
    """Main application logic."""
    st.title("🍳 AI Recipe Recommendation System")
    st.markdown("Built with real-world datasets: 64k Kaggle recipes + Cleaned Indian Recipes Dataset")
    st.markdown("---")
    
    # Load system
    if st.session_state.recommender is None:
        with st.spinner("Initializing recommendation system (Phases 2-4)..."):
            recipes_df = load_system()
            if recipes_df is None:
                st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Your Preferences")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ingredients_input = st.text_area(
            "🥘 Available Ingredients (comma-separated)",
            placeholder="e.g., chicken, onion, garlic, turmeric",
            height=100
        )
    
    with col2:
        meal_type = st.selectbox(
            "🍽️ Meal Type",
            ["All", "breakfast", "lunch", "dinner", "snack", "dessert"]
        )
        
        cuisine = st.selectbox(
            "🌍 Cuisine Preference",
            ["All", "Indian", "Chinese", "Italian", "Mexican", "Thai", 
             "French", "Japanese", "American", "Mediterranean"]
        )
        
        max_cooking_time = st.slider(
            "⏱️ Maximum Cooking Time (minutes)",
            min_value=5,
            max_value=180,
            value=60,
            step=5
        )
    
    num_recommendations = st.sidebar.slider(
        "Number of Recipes",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    # Get recommendations
    if st.button("🔍 Get Recipe Recommendations", use_container_width=True, type="primary"):
        if not ingredients_input.strip():
            st.warning("Please enter at least one ingredient.")
        else:
            with st.spinner("Finding perfect recipes for you..."):
                try:
                    # Get recommendations
                    recommendations = st.session_state.recommender.recommend(
                        user_ingredients=ingredients_input,
                        meal_type=None if meal_type == "All" else meal_type,
                        cuisine=None if cuisine == "All" else cuisine,
                        max_cooking_time=max_cooking_time,
                        n_recommendations=num_recommendations,
                        scorer=st.session_state.scorer
                    )
                    
                    if recommendations.empty:
                        st.info("❌ No recipes found matching your criteria. Try adjusting your preferences!")
                    else:
                        # Check for fallback notification
                        if hasattr(recommendations, '_fallback_notification'):
                            st.warning(f"⚠️ {recommendations._fallback_notification}")
                        
                        st.markdown("## 📋 Recommended Recipes")
                        st.markdown("---")
                        
                        for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.subheader(f"{idx}. {recipe['recipe_name']}")
                                
                                with col2:
                                    st.metric("Match Score", f"{recipe['final_score']:.1%}")
                                
                                # Recipe details
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Cuisine:** {recipe['cuisine']}")
                                with col2:
                                    st.write(f"**Meal Type:** {recipe['meal_type']}")
                                with col3:
                                    st.write(f"**⏱️ Time:** {recipe['cooking_time']} min")
                                
                                # Ingredients overlap info (new)
                                st.write(f"**Ingredient Overlap:** {int(recipe['ingredient_overlap_count'])} of your ingredients matched")
                                
                                # Ingredients
                                st.write("**Ingredients:**")
                                ingredients = recipe['ingredients'].split(", ")
                                for ingredient in ingredients:
                                    st.write(f"• {ingredient}")
                                
                                # Instructions
                                st.write("**Instructions:**")
                                st.write(recipe['instructions'])
                                
                                # YouTube links
                                st.write("**📺 Video Tutorials:**")
                                video_links = st.session_state.youtube_gen.generate_links(
                                    recipe_name=recipe['recipe_name'],
                                    cuisine=recipe['cuisine']
                                )
                                for i, link in enumerate(video_links, 1):
                                    st.markdown(f"[Watch on YouTube - Video {i}]({link})")
                                
                                # Explanation
                                st.write("**Why this recipe?**")
                                explanation = (
                                    f"This recipe matches {recipe['final_score']:.1%} of your preferences. "
                                    f"Your ingredients cover {recipe['ingredient_similarity']:.1%} of the recipe ingredients, "
                                    f"and it fits your desired {meal_type if meal_type != 'All' else 'any meal type'} "
                                    f"in {recipe['cooking_time']} minutes."
                                )
                                st.info(explanation)
                                
                                st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    import traceback
                    st.write(traceback.format_exc())

if __name__ == "__main__":
    main()
