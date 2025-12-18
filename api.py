"""
FastAPI Backend for SHL Recommendation System
Provides /health, /recommend endpoints and serves the Frontend
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
            # Fallback check for Render/Docker env vars
            if "GEMINI_API_KEY" in os.environ:
                 engine.gemini_api_key = os.environ["GEMINI_API_KEY"]
            else:
                 print("⚠️ WARNING: GEMINI_API_KEY not found in .env. Checking system env...")
        
        # Load pre-computed embeddings
        if os.path.exists('shl_embeddings.npy'):
            engine.load_embeddings('shl_embeddings.npy')
            print("✅ Recommendation engine initialized successfully!")
        else:
            print("⚠️ Embeddings file not found. API will fail unless built.")
        
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        # We don't raise here to allow the app to start (and show logs) even if engine fails

# --- MODIFIED ROOT ENDPOINT TO SERVE FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the root directory.</p>", status_code=404)

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
        print(f"Error generating recommendations: {str(e)}")
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