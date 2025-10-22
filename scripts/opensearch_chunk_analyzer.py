#!/usr/bin/env python3
"""
OpenSearch Chunk Analyzer
Analyzes chunks in OpenSearch index and generates comprehensive statistics report.
"""

import sys
import os
import json
import statistics
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re

# Add src to path - handle both direct execution and execution from scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
if not os.path.exists(src_path):
    # If src not found, try parent directory (for scripts/ execution)
    src_path = os.path.join(os.path.dirname(script_dir), 'src')

sys.path.insert(0, os.path.dirname(src_path))

from src.utils.connection_manager import ConnectionManager
from src.utils.multi_app_config_manager import MultiAppConfigManager


class OpenSearchChunkAnalyzer:
    """Analyzes chunks stored in OpenSearch and generates detailed statistics."""
    
    def __init__(self, config_path: str = "config/multi_app_config.yaml", app_name: str = "gadea"):
        """Initialize the analyzer with configuration."""
        self.config_path = config_path
        self.app_name = app_name
        self.config_manager = MultiAppConfigManager(config_path)
        self.config = self.config_manager.create_legacy_config(app_name)
        self.connection_manager = ConnectionManager(config_path=None, config_dict=self.config)
        self.opensearch_client = self.connection_manager.get_opensearch_client()
        self.index_name = self.config['services']['opensearch']['index_name']
        
    def _load_config(self) -> Dict:
        """Load configuration from multi-app config manager."""
        return self.config
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation."""
        if not text:
            return 0
        # Rough estimation: 1 token ≈ 4 characters for English/Spanish text
        return len(text) // 4
    
    def _get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all chunks from OpenSearch index."""
        print("🔍 Retrieving all chunks from OpenSearch...")
        
        chunks = []
        scroll_size = 1000
        
        # Initial search with scroll
        search_body = {
            "size": scroll_size,
            "query": {"match_all": {}},
            "_source": ["content", "title", "file_name", "chunk_id", "metadata", 
                       "document_type", "has_images"]
        }
        
        try:
            response = self.opensearch_client.search(
                index=self.index_name,
                body=search_body,
                scroll='2m'
            )
            
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            chunks.extend(hits)
            
            print(f"📄 Retrieved {len(hits)} chunks in first batch...")
            
            # Continue scrolling
            while hits:
                response = self.opensearch_client.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )
                
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']
                chunks.extend(hits)
                
                if hits:
                    print(f"📄 Retrieved {len(chunks)} chunks total...")
            
            # Clear scroll
            self.opensearch_client.clear_scroll(scroll_id=scroll_id)
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            return []
        
        print(f"✅ Retrieved {len(chunks)} total chunks")
        return chunks
    
    def _analyze_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze chunks and generate comprehensive statistics."""
        print("📊 Analyzing chunk statistics...")
        
        if not chunks:
            return {"error": "No chunks found"}
        
        # Extract content and calculate token counts
        token_counts = []
        char_counts = []
        file_stats = defaultdict(list)
        document_types = defaultdict(int)
        chunks_with_images = 0
        
        for chunk in chunks:
            source = chunk.get('_source', {})
            content = source.get('content', '')
            file_name = source.get('file_name', 'unknown')
            doc_type = source.get('document_type', 'unknown')
            has_images = source.get('has_images', False)
            
            # Calculate tokens and characters
            tokens = self._estimate_tokens(content)
            chars = len(content)
            
            token_counts.append(tokens)
            char_counts.append(chars)
            file_stats[file_name].append(tokens)
            document_types[doc_type] += 1
            
            if has_images:
                chunks_with_images += 1
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "token_statistics": self._calculate_statistics(token_counts, "tokens"),
            "character_statistics": self._calculate_statistics(char_counts, "characters"),
            "file_statistics": self._analyze_file_stats(file_stats),
            "document_type_distribution": dict(document_types),
            "chunks_with_images": chunks_with_images,
            "image_percentage": round((chunks_with_images / len(chunks)) * 100, 2),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return stats
    
    def _calculate_statistics(self, values: List[int], unit: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a list of values."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "mode": statistics.mode(values) if len(set(values)) < len(values) else None,
            "std_dev": round(statistics.stdev(values) if len(values) > 1 else 0, 2),
            "percentiles": {
                "p10": self._percentile(sorted_values, 10),
                "p25": self._percentile(sorted_values, 25),
                "p50": self._percentile(sorted_values, 50),  # median
                "p75": self._percentile(sorted_values, 75),
                "p80": self._percentile(sorted_values, 80),
                "p90": self._percentile(sorted_values, 90),
                "p95": self._percentile(sorted_values, 95),
                "p99": self._percentile(sorted_values, 99)
            },
            "distribution": self._analyze_distribution(values, unit)
        }
    
    def _percentile(self, sorted_values: List[int], percentile: int) -> int:
        """Calculate percentile value."""
        if not sorted_values:
            return 0
        
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return int(sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f]))
        else:
            return sorted_values[f]
    
    def _analyze_distribution(self, values: List[int], unit: str) -> Dict[str, Any]:
        """Analyze distribution of values into ranges."""
        if not values:
            return {}
        
        max_val = max(values)
        
        # Define ranges based on unit
        if unit == "tokens":
            ranges = [
                (0, 500, "Very Small (0-500)"),
                (501, 1000, "Small (501-1000)"),
                (1001, 2000, "Medium (1001-2000)"),
                (2001, 3000, "Large (2001-3000)"),
                (3001, 5000, "Very Large (3001-5000)"),
                (5001, float('inf'), "Huge (5000+)")
            ]
        else:  # characters
            ranges = [
                (0, 2000, "Very Small (0-2K)"),
                (2001, 4000, "Small (2K-4K)"),
                (4001, 8000, "Medium (4K-8K)"),
                (8001, 12000, "Large (8K-12K)"),
                (12001, 20000, "Very Large (12K-20K)"),
                (20001, float('inf'), "Huge (20K+)")
            ]
        
        distribution = {}
        for min_val, max_val, label in ranges:
            count = sum(1 for v in values if min_val <= v <= max_val)
            percentage = round((count / len(values)) * 100, 2)
            distribution[label] = {
                "count": count,
                "percentage": percentage
            }
        
        return distribution
    
    def _analyze_file_stats(self, file_stats: Dict[str, List[int]]) -> Dict[str, Any]:
        """Analyze statistics per file."""
        file_analysis = {}
        
        for file_name, tokens_list in file_stats.items():
            if tokens_list:
                file_analysis[file_name] = {
                    "chunk_count": len(tokens_list),
                    "total_tokens": sum(tokens_list),
                    "avg_tokens_per_chunk": round(statistics.mean(tokens_list), 2),
                    "min_tokens": min(tokens_list),
                    "max_tokens": max(tokens_list)
                }
        
        # Sort by total tokens descending
        sorted_files = sorted(file_analysis.items(), 
                            key=lambda x: x[1]['total_tokens'], 
                            reverse=True)
        
        return {
            "per_file": dict(sorted_files[:20]),  # Top 20 files
            "total_files": len(file_stats),
            "files_with_most_chunks": sorted(file_stats.items(), 
                                           key=lambda x: len(x[1]), 
                                           reverse=True)[:10]
        }
    
    def _generate_caching_analysis(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis specific to prompt caching requirements."""
        token_stats = stats.get('token_statistics', {})
        
        if not token_stats:
            return {}
        
        # Caching thresholds
        min_cache_tokens = 2048
        
        # Calculate chunks suitable for caching
        percentiles = token_stats.get('percentiles', {})
        
        caching_analysis = {
            "min_tokens_for_caching": min_cache_tokens,
            "chunks_above_cache_threshold": {
                "count": sum(1 for chunk in self.chunks_data if chunk >= min_cache_tokens),
                "percentage": 0
            },
            "recommended_chunk_retrieval": {
                "for_2048_tokens": self._calculate_chunks_needed(token_stats, 2048),
                "for_3000_tokens": self._calculate_chunks_needed(token_stats, 3000),
                "for_4000_tokens": self._calculate_chunks_needed(token_stats, 4000)
            },
            "current_config_analysis": self._analyze_current_config(stats)
        }
        
        # Calculate percentage
        total_chunks = stats.get('total_chunks', 0)
        if total_chunks > 0:
            above_threshold = caching_analysis["chunks_above_cache_threshold"]["count"]
            caching_analysis["chunks_above_cache_threshold"]["percentage"] = round(
                (above_threshold / total_chunks) * 100, 2
            )
        
        return caching_analysis
    
    def _calculate_chunks_needed(self, token_stats: Dict[str, Any], target_tokens: int) -> Dict[str, Any]:
        """Calculate how many chunks needed to reach target tokens."""
        mean_tokens = token_stats.get('mean', 0)
        median_tokens = token_stats.get('median', 0)
        p75_tokens = token_stats.get('percentiles', {}).get('p75', 0)
        
        if mean_tokens == 0:
            return {"error": "No token statistics available"}
        
        return {
            "based_on_mean": max(1, int(target_tokens / mean_tokens)),
            "based_on_median": max(1, int(target_tokens / median_tokens)),
            "based_on_p75": max(1, int(target_tokens / p75_tokens)) if p75_tokens > 0 else "N/A",
            "recommended": max(1, int(target_tokens / median_tokens)) + 1  # +1 for safety margin
        }
    
    def _analyze_current_config(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current configuration effectiveness."""
        current_chunk_size = self.config.get('rag_system', {}).get('chunking', {}).get('chunk_size', 1000)
        current_max_results = self.config.get('rag_system', {}).get('search', {}).get('max_results', 5)
        
        token_stats = stats.get('token_statistics', {})
        mean_tokens = token_stats.get('mean', 0)
        
        estimated_context_tokens = current_max_results * mean_tokens
        
        return {
            "configured_chunk_size": current_chunk_size,
            "configured_max_results": current_max_results,
            "estimated_context_tokens": round(estimated_context_tokens, 2),
            "cache_eligible": estimated_context_tokens >= 2048,
            "cache_efficiency_rating": self._rate_cache_efficiency(estimated_context_tokens)
        }
    
    def _rate_cache_efficiency(self, context_tokens: float) -> str:
        """Rate the cache efficiency based on context tokens."""
        if context_tokens < 2048:
            return "❌ Poor - Below caching threshold"
        elif context_tokens < 3000:
            return "⚠️ Fair - Just above threshold"
        elif context_tokens < 4000:
            return "✅ Good - Well above threshold"
        else:
            return "🚀 Excellent - High caching efficiency"
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive chunk analysis report."""
        print("🚀 Starting OpenSearch Chunk Analysis...")
        print(f"📋 Index: {self.index_name}")
        print(f"🔧 Config: {self.config_path}")
        print("-" * 60)
        
        # Get all chunks
        chunks = self._get_all_chunks()
        
        if not chunks:
            print("❌ No chunks found in index")
            return {"error": "No chunks found"}
        
        # Store chunks data for caching analysis
        self.chunks_data = [self._estimate_tokens(chunk.get('_source', {}).get('content', '')) 
                           for chunk in chunks]
        
        # Analyze chunks
        stats = self._analyze_chunks(chunks)
        
        # Add caching analysis
        stats['caching_analysis'] = self._generate_caching_analysis(stats)
        
        # Generate report
        report = {
            "analysis_info": {
                "index_name": self.index_name,
                "config_file": self.config_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_version": "1.0.0"
            },
            "statistics": stats
        }
        
        # Save to file if specified
        if output_file:
            self._save_report(report, output_file)
        
        # Print summary
        self._print_summary(stats)
        
        return report
    
    def _save_report(self, report: Dict[str, Any], output_file: str):
        """Save report to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"💾 Report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def _print_summary(self, stats: Dict[str, Any]):
        """Print summary of analysis results."""
        print("\n" + "=" * 60)
        print("📊 OPENSEARCH CHUNK ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Basic stats
        print(f"📄 Total Chunks: {stats.get('total_chunks', 0):,}")
        
        # Token statistics
        token_stats = stats.get('token_statistics', {})
        if token_stats:
            print(f"\n🔤 TOKEN STATISTICS:")
            print(f"   Mean: {token_stats.get('mean', 0)} tokens")
            print(f"   Median: {token_stats.get('median', 0)} tokens")
            print(f"   Min: {token_stats.get('min', 0)} tokens")
            print(f"   Max: {token_stats.get('max', 0)} tokens")
            print(f"   P80: {token_stats.get('percentiles', {}).get('p80', 0)} tokens")
            print(f"   P95: {token_stats.get('percentiles', {}).get('p95', 0)} tokens")
        
        # Character statistics
        char_stats = stats.get('character_statistics', {})
        if char_stats:
            print(f"\n📝 CHARACTER STATISTICS:")
            print(f"   Mean: {char_stats.get('mean', 0):.0f} characters")
            print(f"   Median: {char_stats.get('median', 0):.0f} characters")
            print(f"   Min: {char_stats.get('min', 0)} characters")
            print(f"   Max: {char_stats.get('max', 0)} characters")
            print(f"   P80: {char_stats.get('percentiles', {}).get('p80', 0)} characters")
            print(f"   P95: {char_stats.get('percentiles', {}).get('p95', 0)} characters")
            
            # Show character-to-token ratio
            if token_stats and token_stats.get('mean', 0) > 0:
                ratio = char_stats.get('mean', 0) / token_stats.get('mean', 1)
                print(f"   Chars/Token Ratio: {ratio:.2f} (expected: ~4.0)")
        
        # Configuration vs Reality
        current_chunk_size = self.config.get('rag_system', {}).get('chunking', {}).get('chunk_size', 1000)
        if char_stats:
            actual_avg = char_stats.get('mean', 0)
            discrepancy = ((current_chunk_size - actual_avg) / current_chunk_size) * 100 if current_chunk_size > 0 else 0
            print(f"\n⚙️  CONFIGURATION vs REALITY:")
            print(f"   Configured chunk_size: {current_chunk_size} characters")
            print(f"   Actual average chunk: {actual_avg:.0f} characters")
            if discrepancy > 10:
                print(f"   Discrepancy: {discrepancy:.1f}% smaller than configured")
        
        # Caching analysis
        caching = stats.get('caching_analysis', {})
        if caching:
            print(f"\n🚀 CACHING ANALYSIS:")
            current_config = caching.get('current_config_analysis', {})
            if current_config:
                print(f"   Current Config: {current_config.get('configured_max_results', 0)} chunks × "
                      f"{current_config.get('estimated_context_tokens', 0):.0f} tokens")
                print(f"   Cache Eligible: {'✅ Yes' if current_config.get('cache_eligible') else '❌ No'}")
                print(f"   Efficiency: {current_config.get('cache_efficiency_rating', 'Unknown')}")
        
        # File stats
        file_stats = stats.get('file_statistics', {})
        if file_stats:
            print(f"\n📁 FILE STATISTICS:")
            print(f"   Total Files: {file_stats.get('total_files', 0)}")
            
            top_files = file_stats.get('files_with_most_chunks', [])[:3]
            if top_files:
                print(f"   Top Files by Chunk Count:")
                for file_name, tokens_list in top_files:
                    print(f"     • {file_name}: {len(tokens_list)} chunks")
        
        # Images
        if stats.get('chunks_with_images', 0) > 0:
            print(f"\n🖼️  MULTIMODAL CONTENT:")
            print(f"   Chunks with Images: {stats.get('chunks_with_images', 0)} "
                  f"({stats.get('image_percentage', 0)}%)")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze OpenSearch chunks and generate statistics report')
    parser.add_argument('--config', '-c', 
                       default='config/multi_app_config.yaml',
                       help='Path to multi-application configuration file')
    parser.add_argument('--app', '-a',
                       default='gadea',
                       help='Application name (gadea, pds, etc.)')
    parser.add_argument('--output', '-o',
                       help='Output file for JSON report')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        analyzer = OpenSearchChunkAnalyzer(args.config, args.app)
        
        # Generate timestamp for default output file
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"opensearch_chunk_analysis_{timestamp}.json"
        
        report = analyzer.generate_report(args.output)
        
        if args.verbose:
            print(f"\n📋 Full report saved to: {args.output}")
            print("🔍 Use 'jq' or any JSON viewer to explore the detailed results")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
