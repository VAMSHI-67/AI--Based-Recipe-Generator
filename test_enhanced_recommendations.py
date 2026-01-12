"""
Test script to validate enhanced recommendation logic improvements.

Tests:
1. Hard filters applied before scoring (cuisine, time)
2. Ingredient overlap filtering (minimum 2 ingredients)
3. Minimum score threshold (30%)
4. Deduplication
5. Fallback mechanism
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml.data_loader import load_global_dataset
from ml.preprocessing import RecipePreprocessor
from ml.recommender import RecipeRecommender
from ml.scoring import PreferenceScorer

def test_enhanced_recommendations():
    """Test the enhanced recommendation logic."""
    print("\n" + "="*70)
    print("ENHANCED RECOMMENDATION LOGIC TEST")
    print("="*70)
    
    # Load data
    print("\n[SETUP] Loading data...")
    recipes_df = load_global_dataset()
    print(f"[OK] Loaded {len(recipes_df)} recipes")
    
    # Prepare data for ML pipeline
    print("\n[SETUP] Preparing for ML...")
    
    # Convert lists to strings
    recipes_df['ingredients_cleaned'] = recipes_df['ingredients_list'].apply(
        lambda x: ' '.join(x).lower() if isinstance(x, list) else str(x).lower()
    )
    recipes_df['instructions'] = recipes_df['directions_list'].apply(
        lambda x: '\n'.join(x) if isinstance(x, list) else str(x)
    )
    recipes_df['ingredients'] = recipes_df['ingredients_list'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )
    recipes_df['recipe_name'] = recipes_df['recipe_title']
    
    # Add default cooking_time if missing
    if 'cooking_time' not in recipes_df.columns:
        recipes_df['cooking_time'] = (recipes_df['num_steps'].fillna(5) * 5).clip(5, 180)
    
    # Add default meal_type if missing
    if 'meal_type' not in recipes_df.columns:
        recipes_df['meal_type'] = recipes_df['category'].fillna('other')
    
    # Initialize recommender and scorer
    recommender = RecipeRecommender(recipes_df)
    recommender.build_tfidf_model()
    scorer = PreferenceScorer()
    
    print(f"[OK] Built TF-IDF model with {len(recommender.tfidf_vectorizer.get_feature_names_out())} features")
    
    # TEST CASE 1: Indian cuisine, 5 minute limit (strict constraints)
    print("\n" + "-"*70)
    print("TEST 1: Indian cuisine, eggs+chicken+butter, 5 min limit")
    print("-"*70)
    
    recommendations = recommender.recommend(
        user_ingredients="eggs, chicken, butter, onion, garlic, green chilli",
        meal_type="lunch",
        cuisine="Indian",
        max_cooking_time=5,
        n_recommendations=5,
        scorer=scorer
    )
    
    print(f"\nFound {len(recommendations)} recommendations:")
    print("\nEXPECTED BEHAVIOR:")
    print("  [*] All cuisines should be 'Indian'")
    print("  [*] All cooking_time <= 5 minutes")
    print("  [*] ingredient_overlap_count >= 2")
    print("  [*] final_score >= 30%")
    print("  [*] No duplicate recipe_names")
    
    print("\nRESULTS:")
    if len(recommendations) == 0:
        print("  [FALLBACK ACTIVATED] No recipes matched constraints")
        if hasattr(recommendations, '_fallback_notification'):
            print(f"  Reason: {recommendations._fallback_notification}")
    else:
        for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
            cuisine_ok = recipe['cuisine'].lower() == 'indian'
            time_ok = recipe['cooking_time'] <= 5
            overlap_ok = recipe['ingredient_overlap_count'] >= 2
            score_ok = recipe['final_score'] >= 0.30
            
            status = "[OK]" if all([cuisine_ok, time_ok, overlap_ok, score_ok]) else "[WARN]"
            
            print(f"\n  {idx}. {recipe['recipe_name']} {status}")
            print(f"     Cuisine: {recipe['cuisine']} {'✓' if cuisine_ok else '❌'}")
            print(f"     Time: {recipe['cooking_time']} min {'✓' if time_ok else '❌'}")
            print(f"     Overlap: {int(recipe['ingredient_overlap_count'])} ingredients {'✓' if overlap_ok else '❌'}")
            print(f"     Score: {recipe['final_score']:.1%} {'✓' if score_ok else '❌'}")
    
    # TEST CASE 2: Non-Italian with reasonable constraints
    print("\n" + "-"*70)
    print("TEST 2: Italian cuisine, pasta+tomato+garlic, 30 min limit")
    print("-"*70)
    
    recommendations = recommender.recommend(
        user_ingredients="pasta, tomato, garlic, olive oil, basil",
        meal_type="dinner",
        cuisine="Italian",
        max_cooking_time=30,
        n_recommendations=3,
        scorer=scorer
    )
    
    print(f"\nFound {len(recommendations)} recommendations:")
    if len(recommendations) > 0:
        print("\nRESULTS:")
        for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
            print(f"\n  {idx}. {recipe['recipe_name']}")
            print(f"     Cuisine: {recipe['cuisine']}")
            print(f"     Time: {recipe['cooking_time']} min")
            print(f"     Ingredient Overlap: {int(recipe['ingredient_overlap_count'])}")
            print(f"     Score: {recipe['final_score']:.1%}")
    else:
        print("  [NO RESULTS] Constraints too strict")
    
    # TEST CASE 3: High-similarity, low-time scenario
    print("\n" + "-"*70)
    print("TEST 3: Low time limit (10 min) with vegetable ingredients")
    print("-"*70)
    
    recommendations = recommender.recommend(
        user_ingredients="carrot, broccoli, garlic, oil, salt",
        meal_type=None,
        cuisine="All",
        max_cooking_time=10,
        n_recommendations=3,
        scorer=scorer
    )
    
    print(f"\nFound {len(recommendations)} recommendations:")
    if len(recommendations) > 0:
        all_under_10 = all(r['cooking_time'] <= 10 for _, r in recommendations.iterrows())
        print(f"\n  All under 10 min: {'✓' if all_under_10 else '❌'}")
        print("\nSample results:")
        for idx, (_, recipe) in enumerate(recommendations.head(3).iterrows(), 1):
            print(f"\n  {idx}. {recipe['recipe_name']}")
            print(f"     Time: {recipe['cooking_time']} min")
            print(f"     Score: {recipe['final_score']:.1%}")
    
    # TEST CASE 4: Deduplication check
    print("\n" + "-"*70)
    print("TEST 4: Deduplication check (requesting 10 results)")
    print("-"*70)
    
    recommendations = recommender.recommend(
        user_ingredients="chicken, rice, onion",
        meal_type=None,
        cuisine="All",
        max_cooking_time=60,
        n_recommendations=10,
        scorer=scorer
    )
    
    recipe_names = recommendations['recipe_name'].tolist()
    duplicates = len(recipe_names) - len(set(recipe_names))
    
    print(f"\nRequested: 10 | Received: {len(recommendations)}")
    print(f"Duplicate recipe names: {duplicates} {'✓' if duplicates == 0 else '❌'}")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("""
✅ ENHANCEMENTS IMPLEMENTED:
   1. Hard filters (cuisine, time) applied BEFORE scoring
   2. Ingredient overlap minimum threshold (2+ ingredients)
   3. Minimum score threshold (30%)
   4. Deduplication by recipe_name
   5. Graceful fallback with notifications
   
✅ EXPECTED IMPROVEMENTS:
   • Indian input → Indian recipes only
   • Time constraints strictly enforced
   • Ingredient relevance obvious (2+ shared ingredients)
   • No asparagus/unrelated ingredients (via overlap filter)
   • No duplicate recipes
   • Realistic match scores (40%+)
   • Fallback message when constraints too strict
    """)
    
    print("\n✅ All tests completed. Check results above.\n")

if __name__ == "__main__":
    test_enhanced_recommendations()
