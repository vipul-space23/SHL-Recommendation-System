"""
evaluation.py - EMERGENCY VERSION
Hardcoded API Key + Excel Fix
"""

import os
import pandas as pd
import time
from typing import List, Dict

# ==========================================
# 🛑 PASTE YOUR FRESH API KEY HERE DIRECTLY
# ==========================================
MY_API_KEY = "AIzaSyAQSrrwKZlPNxeTRPNippU0ToWFIPzdeak"
# ==========================================


# Import the recommendation engine
try:
    from shl_recommendation_engine import SHLRecommendationEngine
except ImportError:
    try:
        from recommendation_engine import SHLRecommendationEngine
    except ImportError:
        print("❌ Error: Cannot import SHLRecommendationEngine")
        exit(1)

def read_data_file(filepath):
    """
    Forcefully reads the file even if extension is wrong.
    Handles 'test.csv' actually being an Excel file.
    """
    print(f"   📂 Reading {filepath}...")
    
    # 1. Try reading as Excel (Fixes the garbage text issue)
    try:
        df = pd.read_excel(filepath)
        print("   ✅ Detected Excel format (even if named .csv)")
        return df
    except Exception:
        pass 

    # 2. Try reading as CSV
    try:
        return pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', engine='python')
    except Exception:
        return pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip', engine='python')

def get_query_column(df):
    """Find the query column"""
    possible_names = ['query', 'question', 'prompt', 'jd', 'description']
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    # Fallback
    return df.columns[0]

def generate_test_predictions(engine: SHLRecommendationEngine, 
                              test_file: str = 'test.csv',
                              output_file: str = 'submission.csv'):
    
    print(f"\n🔮 Generating test predictions...")
    
    if not os.path.exists(test_file):
        print(f"❌ Error: {test_file} not found.")
        return

    # READ FILE
    test_df = read_data_file(test_file)
    
    # IDENTIFY COLUMN
    query_col = get_query_column(test_df)
    print(f"   ✅ Using column: {query_col}")

    # PREVIEW DATA (Sanity Check)
    first_val = str(test_df[query_col].iloc[0])
    if len(first_val) > 100:
        print(f"   👀 First query: {first_val[:50]}...")
    else:
        print(f"   👀 First query: {first_val}")

    # START PROCESSING
    queries = test_df[query_col].unique()
    predictions_data = []
    
    print(f"   Processing {len(queries)} unique queries...")

    for i, query in enumerate(queries, 1):
        if pd.isna(query) or str(query).strip() == "":
            continue
            
        try:
            # CALL ENGINE
            recommendations = engine.recommend_balanced(str(query), top_k=10)
            
            for rec in recommendations:
                predictions_data.append({
                    "Query": query,
                    "Assessment_url": rec['url']
                })
            
            if i % 5 == 0:
                print(f"   ✓ Processed {i}/{len(queries)}")
                
        except Exception as e:
            print(f"   ❌ Error on query {i}: {e}")
            # If API fails, we MUST stop to notify user
            if "400" in str(e) or "API key" in str(e):
                print("\n🛑 STOP: API Key rejected. Check the hardcoded key.")
                return
    
    # SAVE
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(output_file, index=False)
    print("\n" + "="*60)
    print(f"✅ DONE! Saved to {output_file}")
    print("="*60)

def main():
    print("🚀 Initializing...")
    
    # INITIALIZE ENGINE WITH HARDCODED KEY
    engine = SHLRecommendationEngine('shl_catalog.json')
    engine.gemini_api_key = MY_API_KEY  # <--- OVERRIDE WITH HARDCODED KEY
    
    if not engine.gemini_api_key or "PASTE" in engine.gemini_api_key:
        print("❌ CRITICAL: You forgot to paste your API key in the script!")
        return

    # LOAD EMBEDDINGS
    if os.path.exists('shl_embeddings.npy'):
        engine.load_embeddings('shl_embeddings.npy')
    else:
        print("⚠️ Embeddings missing. Building now...")
        engine.build_embeddings()
    
    # RUN
    generate_test_predictions(engine, 'test.csv', 'submission.csv')

if __name__ == "__main__":
    main()