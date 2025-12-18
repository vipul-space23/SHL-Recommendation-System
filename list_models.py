"""
List all available Gemini models for your API key
"""

import requests
import json

print("🔍 LISTING AVAILABLE GEMINI MODELS")
print("="*70)
print()

# Read .env file DIRECTLY
try:
    with open('.env', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the API key
    api_key = None
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('GEMINI_API_KEY'):
            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        exit(1)
    
    print(f"✅ Using API Key: {api_key[:20]}...{api_key[-4:]}")
    print()
    
except FileNotFoundError:
    print("❌ .env file not found")
    exit(1)

# List all models
print("Fetching available models...")
print("-"*70)

url = f"https://generativelanguage.googleapis.com/v1beta/models"

try:
    response = requests.get(
        f"{url}?key={api_key}",
        timeout=15
    )
    
    print(f"Status Code: {response.status_code}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        models = result.get('models', [])
        
        print(f"✅ Found {len(models)} available models:")
        print()
        
        # Separate by type
        text_models = []
        embedding_models = []
        other_models = []
        
        for model in models:
            name = model.get('name', '')
            display_name = model.get('displayName', '')
            supported = model.get('supportedGenerationMethods', [])
            
            if 'embedding' in name.lower() or 'embedContent' in supported:
                embedding_models.append((name, display_name, supported))
            elif 'generateContent' in supported:
                text_models.append((name, display_name, supported))
            else:
                other_models.append((name, display_name, supported))
        
        print("📝 TEXT GENERATION MODELS:")
        print("-"*70)
        for name, display, methods in text_models:
            print(f"  {name}")
            print(f"    Display: {display}")
            print(f"    Methods: {', '.join(methods)}")
            print()
        
        print("🔢 EMBEDDING MODELS:")
        print("-"*70)
        for name, display, methods in embedding_models:
            print(f"  {name}")
            print(f"    Display: {display}")
            print(f"    Methods: {', '.join(methods)}")
            print()
        
        if other_models:
            print("🔧 OTHER MODELS:")
            print("-"*70)
            for name, display, methods in other_models:
                print(f"  {name}")
                print(f"    Display: {display}")
                print(f"    Methods: {', '.join(methods)}")
                print()
        
        print("="*70)
        print("💡 RECOMMENDATION FOR YOUR PROJECT:")
        print("="*70)
        print()
        
        # Find the embedding model
        embedding_model = None
        for name, display, methods in embedding_models:
            if 'embedding' in name.lower():
                embedding_model = name
                break
        
        if embedding_model:
            print(f"✅ Use this for embeddings:")
            print(f"   {embedding_model}")
            print()
            print("Your recommendation_engine.py should use:")
            print(f'   url = "https://generativelanguage.googleapis.com/v1beta/{embedding_model}:embedContent"')
        else:
            print("⚠️  No embedding model found. This might be an API access issue.")
        
    else:
        error = response.json()
        print("❌ Error:")
        print(json.dumps(error, indent=2))
        
        if response.status_code == 400 and "expired" in str(error):
            print()
            print("Your API key is expired. Create a new one at:")
            print("https://aistudio.google.com/app/apikey")
        elif response.status_code == 403:
            print()
            print("API not enabled. Enable it at:")
            print("https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print()