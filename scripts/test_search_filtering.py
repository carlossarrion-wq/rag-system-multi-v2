#!/usr/bin/env python3
"""
Test script to verify stop words filtering in vector database searches
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.hybrid_retriever_fixed import HybridRetrieverFixed
from loguru import logger

def test_search_filtering():
    """Test the stop words filtering in search queries"""
    
    print("🔍 Testing Stop Words Filtering in Vector Database Searches")
    print("=" * 60)
    
    try:
        # Initialize retriever
        retriever = HybridRetrieverFixed("config/multi_app_config.yaml")
        
        # Test queries with stop words
        test_queries = [
            "what is the problem with the system",
            "how to fix this issue",
            "where are the configuration files",
            "when was this document created",
            "why is the application failing",
            "problema con el sistema de configuración",
            "cómo resolver este error en la aplicación"
        ]
        
        print("\n📝 Testing Query Filtering:")
        print("-" * 40)
        
        for query in test_queries:
            try:
                # Test the filtering method directly
                filtered_query = retriever._filter_search_query(query)
                print(f"Original:  '{query}'")
                print(f"Filtered:  '{filtered_query}'")
                print()
                
            except Exception as e:
                print(f"❌ Error filtering query '{query}': {e}")
                print()
        
        print("\n🔍 Testing Actual Search (first query only):")
        print("-" * 40)
        
        # Test actual search with the first query
        test_query = test_queries[0]
        print(f"Searching for: '{test_query}'")
        
        results = retriever.search(test_query, top_k=3)
        
        if results:
            print(f"✅ Search successful! Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('title', 'No title')} (score: {result.get('rrf_score', 0):.4f})")
        else:
            print("⚠️  No results found")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_filtering()
