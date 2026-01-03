"""
AI-Based Recipe Recommendation System
Main Streamlit application entrypoint.

This module loads the recipe dataset and provides an interactive interface
for users to get recipe recommendations based on their preferences.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

def load_system():
    """Load and initialize the recommendation system."""
    try:
        # Load data
        st.session_state.preprocessor = RecipePreprocessor()
        recipes_df = st.session_state.preprocessor.load_and_preprocess("data/recipes.csv")
        
        # Initialize recommender
        st.session_state.recommender = RecipeRecommender(recipes_df)
        st.session_state.recommender.build_tfidf_model()
        
        # Initialize scorer
        st.session_state.scorer = PreferenceScorer()
        
        # Initialize YouTube generator
        st.session_state.youtube_gen = YouTubeRecommendationGenerator()
        
        return recipes_df
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None

def main():
    """Main application logic."""
    st.title("🍳 AI Recipe Recommendation System")
    st.markdown("---")
    
    # Load system
    if st.session_state.recommender is None:
        with st.spinner("Loading recipe database and ML models..."):
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
                        st.info("No recipes found matching your criteria. Try adjusting your preferences!")
                    else:
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

if __name__ == "__main__":
    main()
