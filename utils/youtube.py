"""
YouTube video recommendation generator.

Generates search links for recipe videos without using YouTube API.
Creates properly formatted YouTube search URLs based on recipe name and cuisine.
"""

import urllib.parse
from typing import List


class YouTubeRecommendationGenerator:
    """Generates YouTube search links for recipes."""
    
    SEARCH_BASE_URL = "https://www.youtube.com/results?search_query="
    
    def __init__(self):
        """Initialize YouTube recommendation generator."""
        pass
    
    def generate_links(
        self,
        recipe_name: str,
        cuisine: str = "",
        num_links: int = 2
    ) -> List[str]:
        """
        Generate YouTube search links for a recipe.
        
        Creates multiple search query variations to improve discovery:
        1. Recipe name + cuisine + "recipe"
        2. Recipe name + "how to make"
        3. Recipe name + "cooking tutorial"
        
        Args:
            recipe_name: Name of the recipe
            cuisine: Optional cuisine type
            num_links: Number of links to generate (default 2)
        
        Returns:
            List of YouTube search URLs
        """
        if not recipe_name or not recipe_name.strip():
            return []
        
        # Prepare variations
        variations = []
        
        # Variation 1: Recipe name + cuisine + recipe + step by step
        if cuisine and cuisine.lower() != 'other':
            query1 = f"{recipe_name} {cuisine} recipe step by step"
            variations.append(self._create_search_url(query1))
        
        # Variation 2: Recipe name + how to make
        query2 = f"how to make {recipe_name} easy recipe"
        variations.append(self._create_search_url(query2))
        
        # Variation 3: Recipe name + cooking tutorial
        if len(variations) < num_links:
            query3 = f"{recipe_name} cooking tutorial"
            variations.append(self._create_search_url(query3))
        
        # Variation 4: Just recipe name
        if len(variations) < num_links:
            query4 = f"{recipe_name} recipe"
            variations.append(self._create_search_url(query4))
        
        return variations[:num_links]
    
    def _create_search_url(self, query: str) -> str:
        """
        Create a properly encoded YouTube search URL.
        
        Args:
            query: Search query string
        
        Returns:
            Properly encoded YouTube search URL
        """
        # Clean and normalize query
        query = query.strip().replace(' ', '+')
        
        # URL encode special characters
        query_encoded = urllib.parse.quote(query, safe='+')
        
        return f"{self.SEARCH_BASE_URL}{query_encoded}"
