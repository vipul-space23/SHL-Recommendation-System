"""
Quick script to build embeddings
Run this: python build_embeddings.py
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import recommendation engine with fallback options
try:
    from shl_recommendation_engine import SHLRecommendationEngine
except ImportError:
    try:
        from recommendation_engine import SHLRecommendationEngine
    except ImportError:
        print("❌ ERROR: Cannot import SHLRecommendationEngine")
        print("\n📝 Make sure your recommendation engine file is named:")
        print("   - shl_recommendation_engine.py OR")
        print("   - recommendation_engine.py")
        print("\n💡 Rename your file using:")
        print("   mv your_file.py recommendation_engine.py")
        exit(1)

print("🚀 Building Embeddings...")
print("="*60)

# Initialize engine (API key loads automatically from .env)
engine = SHLRecommendationEngine('shl_catalog.json')

# Check if API key was loaded
if not engine.gemini_api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file")
    print("\n📝 Create a .env file in your project folder with:")
    print("   GEMINI_API_KEY=your_actual_api_key_here")
    print("\n🔑 Get your free API key from: https://ai.google.dev/")
    exit(1)

print(f"✅ API key loaded from .env file")

# Build embeddings
print(f"\n📊 Found {len(engine.catalog)} assessments")
print("⏳ Building embeddings (this takes ~15-20 mins)...\n")

try:
    engine.build_embeddings()
    print("\n" + "="*60)
    print("✅ SUCCESS! Embeddings saved to: shl_embeddings.npy")
    print("="*60)
    print("\n🎯 Next steps:")
    print("  1. Run API server:")
    print("     uvicorn api:app --reload")
    print("\n  2. Test frontend:")
    print("     open index.html (or use a local server)")
    print("\n  3. Generate predictions:")
    print("     python evaluation_script.py")
    print("\n  4. Or run evaluations:")
    print("     python evaluation.py")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Process interrupted by user")
    print("You can resume by running this script again")
    
except Exception as e:
    print("\n" + "="*60)
    print(f"❌ ERROR: {e}")
    print("="*60)
    print("\n🔧 Troubleshooting:")
    print("  1. Check your internet connection")
    print("  2. Verify GEMINI_API_KEY is correct in .env file")
    print("  3. Check API quota at: https://ai.google.dev/")
    print("  4. Make sure shl_catalog.json exists")
    print("  5. Try running with a smaller test first")
    print("\n💡 If quota exceeded, wait 1 minute and try again")
    exit(1)