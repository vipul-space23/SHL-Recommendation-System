"""
FastAPI Backend for SHL Recommendation System
Provides /health and /recommend endpoints
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import recommendation engine
try:
    from shl_recommendation_engine import SHLRecommendationEngine
except ImportError:
    try:
        from recommendation_engine import SHLRecommendationEngine
    except ImportError:
        print("❌ Error: Cannot import SHLRecommendationEngine")
        print("Make sure your recommendation engine file is named:")
        print("  - shl_recommendation_engine.py OR")
        print("  - recommendation_engine.py")
        exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine (global)
engine = None

# Pydantic models
class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class AssessmentRecommendation(BaseModel):
    name: str
    url: str
    test_type: List[str]
    description: str
    duration: int
    remote_testing: str
    adaptive: str

class RecommendResponse(BaseModel):
    recommendations: List[AssessmentRecommendation]
    query: str
    count: int

class HealthResponse(BaseModel):
    status: str
    message: str

# Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup"""
    global engine
    
    print("🚀 Starting SHL Recommendation API...")
    
    try:
        # Initialize engine (API key loads from .env automatically)
        engine = SHLRecommendationEngine('shl_catalog.json')
        
        # Check if API key was loaded
        if not engine.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Load pre-computed embeddings
        engine.load_embeddings('shl_embeddings.npy')
        
        print("✅ Recommendation engine initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "SHL Assessment Recommendation API is running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "success",
        "message": "API is healthy and running"
    }

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    """
    Recommend assessments based on query
    
    Args:
        query: Natural language query or job description
        top_k: Number of recommendations (5-10)
    
    Returns:
        List of recommended assessments
    """
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get recommendations
        top_k = max(5, min(10, request.top_k))
        recommendations = engine.recommend_balanced(request.query, top_k=top_k)
        
        # Format response
        formatted_recs = [
            AssessmentRecommendation(
                name=rec['name'],
                url=rec['url'],
                test_type=rec['test_type'],
                description=rec['description'],
                duration=rec['duration'],
                remote_testing=rec['remote_testing'],
                adaptive=rec['adaptive']
            )
            for rec in recommendations
        ]
        
        return RecommendResponse(
            recommendations=formatted_recs,
            query=request.query,
            count=len(formatted_recs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    global engine
    
    if engine is None:
        return {"error": "Engine not initialized"}
    
    return {
        "total_assessments": len(engine.catalog),
        "embeddings_shape": engine.embeddings.shape if engine.embeddings is not None else None,
        "status": "ready"
    }

# Run server
if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )