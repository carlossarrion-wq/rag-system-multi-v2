#!/usr/bin/env python3
"""
Script to test image size by simulating the retrieval process locally
This will help us understand the size of base64 images being sent to Claude
"""

import sys
import os
import json
import base64
import requests
from datetime import datetime

def simulate_query_and_check_sizes():
    """Simulate the query to check image sizes"""
    
    print("🔍 Simulating query: 'describe el diagrama del usuario 900'")
    print("📡 Connecting to RAG system...")
    
    # Use the multi-app chat endpoint
    url = "http://52.18.245.120:8000/chat"
    
    payload = {
        "query": "describe el diagrama del usuario 900",
        "app": "pds"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🚀 Sending request to RAG system...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful")
            
            # Check if there's information about images in the response
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"📊 Metadata available: {list(metadata.keys())}")
                
                # Look for image information
                if 'sources' in metadata:
                    sources = metadata['sources']
                    print(f"📄 Found {len(sources)} sources")
                    
                    total_image_size = 0
                    image_count = 0
                    
                    for i, source in enumerate(sources):
                        print(f"\n📄 Source {i+1}:")
                        print(f"   Title: {source.get('title', 'N/A')}")
                        print(f"   Filename: {source.get('filename', 'N/A')}")
                        
                        # Check for image data
                        if 'image_data' in source:
                            image_data = source['image_data']
                            if image_data:
                                image_count += 1
                                image_size = len(str(image_data))
                                total_image_size += image_size
                                
                                print(f"   🖼️  Image data size: {image_size:,} characters")
                                print(f"   📏 Estimated bytes: {image_size * 3 // 4:,} bytes")
                                print(f"   📐 Size in MB: {(image_size * 3 // 4) / (1024*1024):.2f} MB")
                                
                                # Estimate tokens (rough calculation)
                                estimated_tokens = image_size // 4
                                print(f"   🔢 Estimated tokens: {estimated_tokens:,}")
                        else:
                            print(f"   📝 No image data")
                    
                    if image_count > 0:
                        print(f"\n📊 TOTAL IMAGE ANALYSIS:")
                        print(f"   Images found: {image_count}")
                        print(f"   Total image data: {total_image_size:,} characters")
                        print(f"   Total estimated bytes: {total_image_size * 3 // 4:,} bytes")
                        print(f"   Total size in MB: {(total_image_size * 3 // 4) / (1024*1024):.2f} MB")
                        
                        # Token estimation
                        total_tokens = total_image_size // 4
                        print(f"   Total estimated tokens: {total_tokens:,}")
                        print(f"   Claude 3 Haiku limit: 200,000 tokens")
                        
                        if total_tokens > 180000:
                            print(f"   ⚠️  WARNING: Likely exceeds token limit!")
                            print(f"   💡 Recommendation: Reduce image size or use fewer images")
                        else:
                            print(f"   ✅ Within token limit")
                    else:
                        print(f"\n📊 No images found in sources")
            
            # Check the actual response
            if 'answer' in result:
                answer = result['answer']
                print(f"\n📝 Response received: {len(answer)} characters")
                if "error" in answer.lower():
                    print(f"❌ Error in response: {answer[:200]}...")
                else:
                    print(f"✅ Successful response: {answer[:100]}...")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this might indicate the system is processing large images")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - check if the RAG system is running")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_local_logs():
    """Check local logs for image size information"""
    print("\n🔍 Checking local logs for image size information...")
    
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]
        log_files.sort(reverse=True)  # Most recent first
        
        print(f"📄 Found {len(log_files)} log files")
        
        # Check the most recent logs
        for log_file in log_files[:5]:  # Check last 5 logs
            log_path = os.path.join(logs_dir, log_file)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for image-related information
                    if 'image' in content.lower() or 'base64' in content.lower():
                        print(f"\n📄 Log file: {log_file}")
                        
                        # Look for specific patterns
                        lines = content.split('\n')
                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['image data', 'base64', 'image size', 'too long']):
                                print(f"   📝 {line.strip()}")
                        
            except Exception as e:
                print(f"❌ Error reading {log_file}: {e}")
    else:
        print("📁 No logs directory found")

if __name__ == "__main__":
    print("🔍 INVESTIGATING BASE64 IMAGE SIZES")
    print("=" * 50)
    
    simulate_query_and_check_sizes()
    check_local_logs()
    
    print("\n🏁 Investigation complete!")
