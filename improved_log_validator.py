"""
Improved Log Validator for LLM Response Logs
Properly extracts JSON from log files with metadata and headers
"""

import os
import json
import sys
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the src directory to the path so we can import the parser
sys.path.append('src')

from generation.structured_response_parser import StructuredResponseParser


class ImprovedLogValidator:
    """Improved validator that properly handles log file format."""
    
    def __init__(self, logs_directory: str = "logs"):
        self.logs_dir = Path(logs_directory)
        self.parser = StructuredResponseParser()
        self.results = {
            'total_files': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'fallback_parses': 0,
            'json_extraction_success': 0,
            'json_extraction_failed': 0,
            'detailed_results': []
        }
    
    def extract_json_from_log(self, content: str) -> Tuple[str, str]:
        """
        Extract JSON from log file content.
        
        Returns:
            Tuple of (extracted_json, extraction_method)
        """
        
        # Method 1: Look for JSON in code blocks
        json_block_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip(), "code_block"
        
        # Method 2: Look for JSON between RAW LLM RESPONSE and RESPONSE ANALYSIS
        raw_response_pattern = r'=== RAW LLM RESPONSE ===\s*(.*?)\s*=== RESPONSE ANALYSIS ==='
        match = re.search(raw_response_pattern, content, re.DOTALL)
        if match:
            raw_content = match.group(1).strip()
            
            # Check if it starts with a JSON block
            if raw_content.startswith('```json'):
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
                if json_match:
                    return json_match.group(1).strip(), "raw_section_code_block"
            
            # Check if it's direct JSON (starts with {)
            if raw_content.startswith('{'):
                # Find the end of the JSON by counting braces
                brace_count = 0
                json_end = 0
                for i, char in enumerate(raw_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    return raw_content[:json_end], "raw_section_direct"
            
            # If there's text before JSON, try to find JSON starting point
            json_start = raw_content.find('{')
            if json_start != -1:
                potential_json = raw_content[json_start:]
                brace_count = 0
                json_end = 0
                for i, char in enumerate(potential_json):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    return potential_json[:json_end], "raw_section_extracted"
        
        # Method 3: Look for any JSON-like structure in the entire content
        json_pattern = r'(\{[^{}]*"response_type"[^{}]*\{.*?\}[^{}]*\})'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(1), "pattern_match"
        
        # Method 4: Find the largest JSON object in the content
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_objects:
            # Return the longest one (likely the most complete)
            longest_json = max(json_objects, key=len)
            return longest_json, "largest_object"
        
        return "", "not_found"
    
    def validate_all_logs(self) -> Dict[str, Any]:
        """Validate all log files with improved JSON extraction."""
        
        print("=== IMPROVED LOG VALIDATION ===")
        print(f"Scanning directory: {self.logs_dir}")
        print()
        
        # Get all log files
        log_files = list(self.logs_dir.glob("llm_response_*.txt"))
        log_files.sort()
        
        self.results['total_files'] = len(log_files)
        
        print(f"Found {len(log_files)} log files to validate")
        print("-" * 70)
        
        for i, log_file in enumerate(log_files, 1):
            print(f"[{i:2d}/{len(log_files)}] {log_file.name}")
            result = self._validate_single_log(log_file)
            self.results['detailed_results'].append(result)
            
            # Update counters
            if result['json_extracted']:
                self.results['json_extraction_success'] += 1
            else:
                self.results['json_extraction_failed'] += 1
            
            if result['status'] == 'success':
                self.results['successful_parses'] += 1
            elif result['status'] == 'fallback':
                self.results['fallback_parses'] += 1
            else:
                self.results['failed_parses'] += 1
            
            # Show progress
            self._print_result_summary(result)
        
        print("-" * 70)
        self._print_final_summary()
        
        return self.results
    
    def _validate_single_log(self, log_file: Path) -> Dict[str, Any]:
        """Validate a single log file with improved JSON extraction."""
        
        result = {
            'filename': log_file.name,
            'status': 'unknown',
            'is_structured': False,
            'json_extracted': False,
            'extraction_method': 'none',
            'error': None,
            'content_length': 0,
            'json_length': 0,
            'json_issues': [],
            'parsed_keys': [],
            'confidence_score': None
        }
        
        try:
            # Read the log file
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            result['content_length'] = len(content)
            
            if not content:
                result['status'] = 'empty'
                result['error'] = 'File is empty'
                return result
            
            # Extract JSON from log content
            extracted_json, extraction_method = self.extract_json_from_log(content)
            result['extraction_method'] = extraction_method
            
            if not extracted_json:
                result['status'] = 'no_json'
                result['error'] = 'Could not extract JSON from log file'
                return result
            
            result['json_extracted'] = True
            result['json_length'] = len(extracted_json)
            
            # Validate the extracted JSON
            self._validate_extracted_json(extracted_json, result)
            
            # Try to parse with the structured response parser
            parsed_data, is_structured = self.parser.parse_response(extracted_json)
            
            result['is_structured'] = is_structured
            result['parsed_keys'] = list(parsed_data.keys()) if isinstance(parsed_data, dict) else []
            
            # Extract confidence score if available
            if 'confidence' in parsed_data and isinstance(parsed_data['confidence'], dict):
                result['confidence_score'] = parsed_data['confidence'].get('score')
            
            # Determine status
            if is_structured:
                result['status'] = 'success'
            else:
                result['status'] = 'fallback'
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _validate_extracted_json(self, json_str: str, result: Dict[str, Any]):
        """Validate the extracted JSON string."""
        
        try:
            parsed = json.loads(json_str)
            result['json_issues'].append('✅ Valid JSON structure')
            
            # Check for expected fields
            if 'response_type' in parsed:
                result['json_issues'].append('✅ Has response_type field')
            if 'answer' in parsed:
                result['json_issues'].append('✅ Has answer field')
            if 'confidence' in parsed:
                result['json_issues'].append('✅ Has confidence field')
            if 'sources' in parsed:
                result['json_issues'].append(f'✅ Has sources field ({len(parsed["sources"])} sources)')
            
        except json.JSONDecodeError as e:
            result['json_issues'].append(f'❌ JSON Error: {e}')
            
            # Check for common issues
            if 'Expecting \',\' delimiter' in str(e):
                result['json_issues'].append('❌ Issue: Unescaped quotes in string values')
            
            if 'Invalid control character' in str(e):
                result['json_issues'].append('❌ Issue: Unescaped control characters')
            
            if 'Expecting property name' in str(e):
                result['json_issues'].append('❌ Issue: Malformed JSON structure')
            
            if 'Unterminated string' in str(e):
                result['json_issues'].append('❌ Issue: Unterminated string literal')
        
        # Check for structural issues
        if json_str.count('{') != json_str.count('}'):
            result['json_issues'].append('❌ Issue: Mismatched braces')
        
        if json_str.rstrip().endswith(','):
            result['json_issues'].append('❌ Issue: Trailing comma')
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print a summary of a single file validation result."""
        
        status_emoji = {
            'success': '✅',
            'fallback': '🟡',
            'error': '❌',
            'empty': '⚪',
            'no_json': '🔍'
        }
        
        emoji = status_emoji.get(result['status'], '❓')
        
        # Show extraction info
        if result['json_extracted']:
            extraction_info = f"JSON: {result['json_length']}chars ({result['extraction_method']})"
        else:
            extraction_info = f"No JSON extracted ({result['extraction_method']})"
        
        print(f"    {emoji} {result['status'].upper()} | {extraction_info}")
        
        if result['status'] == 'success':
            print(f"        Structured: {result['is_structured']} | Keys: {len(result['parsed_keys'])}")
            if result['confidence_score']:
                print(f"        Confidence: {result['confidence_score']:.2f}")
        
        elif result['status'] == 'fallback':
            print(f"        Fallback parsing | Keys: {len(result['parsed_keys'])}")
        
        elif result['status'] == 'error':
            print(f"        Error: {result['error']}")
        
        # Show JSON validation results
        valid_issues = [issue for issue in result['json_issues'] if issue.startswith('✅')]
        invalid_issues = [issue for issue in result['json_issues'] if issue.startswith('❌')]
        
        if valid_issues:
            print(f"        Valid: {len(valid_issues)} checks passed")
        if invalid_issues:
            print(f"        Issues: {len(invalid_issues)} problems found")
        
        print()
    
    def _print_final_summary(self):
        """Print the final validation summary."""
        
        total = self.results['total_files']
        success = self.results['successful_parses']
        fallback = self.results['fallback_parses']
        failed = self.results['failed_parses']
        json_extracted = self.results['json_extraction_success']
        
        print("=== IMPROVED VALIDATION SUMMARY ===")
        print(f"Total files processed: {total}")
        print(f"🔍 JSON extraction success: {json_extracted}/{total} ({json_extracted/total*100:.1f}%)")
        print(f"✅ Successful structured parses: {success} ({success/total*100:.1f}%)")
        print(f"🟡 Fallback parses: {fallback} ({fallback/total*100:.1f}%)")
        print(f"❌ Failed parses: {failed} ({failed/total*100:.1f}%)")
        print()
        
        # Calculate overall effectiveness
        effective_parses = success + fallback
        print(f"Overall parser effectiveness: {effective_parses}/{total} ({effective_parses/total*100:.1f}%)")
        
        # Show extraction methods used
        extraction_methods = {}
        for result in self.results['detailed_results']:
            method = result['extraction_method']
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        print("\n=== EXTRACTION METHODS ===")
        for method, count in sorted(extraction_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"{count:3d}x {method}")
        
        # Show most common issues
        all_issues = []
        for result in self.results['detailed_results']:
            all_issues.extend(result['json_issues'])
        
        if all_issues:
            print("\n=== VALIDATION RESULTS ===")
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            for issue, count in sorted_issues[:10]:
                print(f"{count:3d}x {issue}")
    
    def generate_detailed_report(self, output_file: str = "improved_log_validation_report.json"):
        """Generate a detailed JSON report."""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    def test_problematic_files(self, max_files: int = 10):
        """Test the ultimate JSON fixer on problematic files."""
        
        print(f"\n=== TESTING ULTIMATE JSON FIXER ===")
        
        # Get files that had JSON extraction but parsing issues
        problematic_files = []
        for result in self.results['detailed_results']:
            if result['json_extracted'] and result['status'] != 'success':
                problematic_files.append(result)
        
        if not problematic_files:
            print("No problematic files found - all extracted JSONs parsed successfully!")
            return
        
        print(f"Testing ultimate fixer on {min(len(problematic_files), max_files)} problematic files...")
        
        improvements = 0
        for result in problematic_files[:max_files]:
            filename = result['filename']
            print(f"\nTesting: {filename}")
            
            # Read and extract JSON from the file
            log_file = self.logs_dir / filename
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            extracted_json, _ = self.extract_json_from_log(content)
            
            if not extracted_json:
                print(f"  ❌ Could not extract JSON")
                continue
            
            # Test with ultimate JSON fix
            try:
                fixed_content = self.parser._ultimate_json_fix(extracted_json)
                parsed = json.loads(fixed_content)
                print(f"  ✅ Ultimate fix successful!")
                print(f"      Keys: {list(parsed.keys())}")
                improvements += 1
            except Exception as e:
                print(f"  ❌ Ultimate fix failed: {e}")
        
        print(f"\nImprovement rate: {improvements}/{min(len(problematic_files), max_files)} files fixed")


def main():
    """Main function to run the improved log validation."""
    
    # Check if logs directory exists
    if not os.path.exists("logs"):
        print("❌ Logs directory not found!")
        return
    
    # Create validator and run validation
    validator = ImprovedLogValidator()
    results = validator.validate_all_logs()
    
    # Generate detailed report
    validator.generate_detailed_report()
    
    # Test ultimate fixer on problematic files
    validator.test_problematic_files()
    
    return results


if __name__ == "__main__":
    main()
