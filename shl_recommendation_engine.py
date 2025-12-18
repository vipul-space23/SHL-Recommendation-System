"""shl_recommendation_engine.py"""

from dotenv import load_dotenv
import os
import json
import numpy as np
from typing import List, Dict
import requests

# Load environment variables from .env file
load_dotenv()

"""
SHL Assessment Recommendation Engine
Uses embeddings + semantic search for recommendations
"""

class SHLRecommendationEngine:
    def __init__(self, catalog_path='shl_catalog.json'):
        """Initialize the recommendation engine"""
        self.catalog = self.load_catalog(catalog_path)
        self.embeddings = None
        # Get API key from environment variable
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            print("⚠️  Warning: GEMINI_API_KEY not found in .env file")
        else:
            print("✅ Gemini API key loaded from .env file")
        
    def load_catalog(self, path):
        """Load the scraped catalog"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def set_api_key(self, api_key):
        """Set Gemini API key (optional - overrides .env)"""
        self.gemini_api_key = api_key
    
    def create_search_text(self, assessment):
        """Create searchable text from assessment data"""
        # Combine all relevant fields for better matching
        test_types = ', '.join(assessment.get('test_type', []))
        
        search_text = f"""
        {assessment['name']}
        {assessment.get('description', '')}
        Test Type: {test_types}
        Remote Testing: {assessment.get('remote_testing', 'No')}
        Duration: {assessment.get('duration', 15)} minutes
        """.strip()
        
        return search_text
    
    def get_embedding_gemini(self, text):
        """Get embedding using Gemini API"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        }
        
        response = requests.post(
            f"{url}?key={self.gemini_api_key}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return np.array(result['embedding']['values'])
        else:
            raise Exception(f"Gemini API error: {response.text}")
    
    def build_embeddings(self):
        """Build embeddings for all assessments"""
        print("🔨 Building embeddings for all assessments...")
        
        if not self.gemini_api_key:
            raise ValueError("Please set Gemini API key in .env file or use set_api_key()")
        
        embeddings = []
        
        for i, assessment in enumerate(self.catalog):
            print(f"  Processing {i+1}/{len(self.catalog)}: {assessment['name'][:50]}...")
            
            search_text = self.create_search_text(assessment)
            embedding = self.get_embedding_gemini(search_text)
            embeddings.append(embedding)
            
            # Rate limiting
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)
        
        self.embeddings = np.array(embeddings)
        
        # Save embeddings
        np.save('shl_embeddings.npy', self.embeddings)
        print(f"✅ Embeddings saved! Shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def load_embeddings(self, path='shl_embeddings.npy'):
        """Load pre-computed embeddings"""
        self.embeddings = np.load(path)
        print(f"✅ Embeddings loaded! Shape: {self.embeddings.shape}")
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Recommend assessments based on query
        
        Args:
            query: Natural language query or job description
            top_k: Number of recommendations (max 10, min 5)
        
        Returns:
            List of recommended assessments
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Run build_embeddings() or load_embeddings() first")
        
        # Ensure top_k is between 5 and 10
        top_k = max(5, min(10, top_k))
        
        print(f"\n🔍 Processing query: {query[:100]}...")
        
        # Get query embedding
        query_embedding = self.get_embedding_gemini(query)
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # Prepare results
        recommendations = []
        for idx in top_indices:
            assessment = self.catalog[idx]
            recommendations.append({
                'name': assessment['name'],
                'url': assessment['url'],
                'test_type': assessment.get('test_type', []),
                'description': assessment.get('description', ''),
                'duration': assessment.get('duration', 15),
                'remote_testing': assessment.get('remote_testing', 'No'),
                'adaptive': assessment.get('adaptive', 'No')
            })
        
        print(f"✅ Found {len(recommendations)} recommendations")
        
        return recommendations
    
    def recommend_balanced(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Recommend with balance between different test types
        Ensures mix of Knowledge/Skills and Personality/Behavior
        """
        # Get initial recommendations (more than needed)
        initial_recs = self.recommend(query, top_k=20)
        
        # Separate by test type
        knowledge_skills = []
        personality_behavior = []
        others = []
        
        for rec in initial_recs:
            test_types = rec['test_type']
            
            if any('Knowledge' in t or 'Skills' in t for t in test_types):
                knowledge_skills.append(rec)
            elif any('Personality' in t or 'Behavior' in t for t in test_types):
                personality_behavior.append(rec)
            else:
                others.append(rec)
        
        # Balance the results
        balanced = []
        
        # Try to get 50-50 split
        target_each = top_k // 2
        
        balanced.extend(knowledge_skills[:target_each])
        balanced.extend(personality_behavior[:target_each])
        
        # Fill remaining with others or extra from main categories
        remaining = top_k - len(balanced)
        if remaining > 0:
            extra = (knowledge_skills[target_each:] + 
                    personality_behavior[target_each:] + 
                    others)
            balanced.extend(extra[:remaining])
        
        return balanced[:top_k]


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize engine (API key loaded automatically from .env)
    engine = SHLRecommendationEngine('shl_catalog.json')
    
    # Build embeddings (run once)
    engine.build_embeddings()
    
    # OR load existing embeddings
    # engine.load_embeddings('shl_embeddings.npy')
    
    # Test recommendation
    query = "I need Java developers who can collaborate with business teams"
    recommendations = engine.recommend_balanced(query, top_k=10)
    
    print("\n" + "="*60)
    print("📋 RECOMMENDATIONS:")
    print("="*60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Types: {', '.join(rec['test_type'])}")
        print(f"   Duration: {rec['duration']} min")
        print(f"   URL: {rec['url']}")