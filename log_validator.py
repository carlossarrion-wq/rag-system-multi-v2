"""
Comprehensive Log Validator for LLM Responses
Validates all log files to ensure structured_response_parser can handle them
"""

import os
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the src directory to the path so we can import the parser
sys.path.append('src')

from generation.structured_response_parser import StructuredResponseParser


class LogValidator:
    """Validates all LLM response logs for JSON parsing compatibility."""
    
    def __init__(self, logs_directory: str = "logs"):
        self.logs_dir = Path(logs_directory)
        self.parser = StructuredResponseParser()
        self.results = {
            'total_files': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'fallback_parses': 0,
            'detailed_results': []
        }
    
    def validate_all_logs(self) -> Dict[str, Any]:
        """Validate all log files in the logs directory."""
        
        print("=== COMPREHENSIVE LOG VALIDATION ===")
        print(f"Scanning directory: {self.logs_dir}")
        print()
        
        # Get all log files
        log_files = list(self.logs_dir.glob("llm_response_*.txt"))
        log_files.sort()
        
        self.results['total_files'] = len(log_files)
        
        print(f"Found {len(log_files)} log files to validate")
        print("-" * 60)
        
        for i, log_file in enumerate(log_files, 1):
            print(f"[{i}/{len(log_files)}] Validating: {log_file.name}")
            result = self._validate_single_log(log_file)
            self.results['detailed_results'].append(result)
            
            # Update counters
            if result['status'] == 'success':
                self.results['successful_parses'] += 1
            elif result['status'] == 'fallback':
                self.results['fallback_parses'] += 1
            else:
                self.results['failed_parses'] += 1
            
            # Show progress
            self._print_result_summary(result)
        
        print("-" * 60)
        self._print_final_summary()
        
        return self.results
    
    def _validate_single_log(self, log_file: Path) -> Dict[str, Any]:
        """Validate a single log file."""
        
        result = {
            'filename': log_file.name,
            'status': 'unknown',
            'is_structured': False,
            'error': None,
            'content_length': 0,
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
            
            # Try to parse with the structured response parser
            parsed_data, is_structured = self.parser.parse_response(content)
            
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
            
            # Additional validation - try to parse as raw JSON
            self._validate_raw_json(content, result)
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _validate_raw_json(self, content: str, result: Dict[str, Any]):
        """Additional validation to check raw JSON parsing issues."""
        
        # Try to parse as raw JSON to identify specific issues
        try:
            json.loads(content)
            result['json_issues'].append('Raw JSON is valid')
        except json.JSONDecodeError as e:
            result['json_issues'].append(f'JSON Error: {e}')
            
            # Check for common issues
            if 'Expecting \',\' delimiter' in str(e):
                result['json_issues'].append('Issue: Unescaped quotes in string values')
            
            if 'Invalid control character' in str(e):
                result['json_issues'].append('Issue: Unescaped control characters (newlines, tabs)')
            
            if 'Expecting property name' in str(e):
                result['json_issues'].append('Issue: Malformed JSON structure')
            
            if 'Unterminated string' in str(e):
                result['json_issues'].append('Issue: Unterminated string literal')
        
        # Check for structural issues
        if content.count('{') != content.count('}'):
            result['json_issues'].append('Issue: Mismatched braces (incomplete JSON)')
        
        if content.rstrip().endswith(','):
            result['json_issues'].append('Issue: Trailing comma')
        
        # Check for common problematic patterns
        if '"/' in content and not '\\"/' in content:
            result['json_issues'].append('Issue: Likely unescaped quotes around paths/endpoints')
        
        if '\n' in content and '\\n' not in content:
            result['json_issues'].append('Issue: Likely unescaped newlines in strings')
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print a summary of a single file validation result."""
        
        status_emoji = {
            'success': '✅',
            'fallback': '🟡',
            'error': '❌',
            'empty': '⚪'
        }
        
        emoji = status_emoji.get(result['status'], '❓')
        status = result['status'].upper()
        
        print(f"   {emoji} {status}")
        
        if result['status'] == 'success':
            print(f"      Structured: {result['is_structured']}")
            print(f"      Keys: {len(result['parsed_keys'])}")
            if result['confidence_score']:
                print(f"      Confidence: {result['confidence_score']:.2f}")
        
        elif result['status'] == 'fallback':
            print(f"      Used fallback parsing")
            print(f"      Keys: {len(result['parsed_keys'])}")
        
        elif result['status'] == 'error':
            print(f"      Error: {result['error']}")
        
        # Show JSON issues if any
        if result['json_issues']:
            print(f"      JSON Issues: {len(result['json_issues'])}")
            for issue in result['json_issues'][:2]:  # Show first 2 issues
                print(f"        - {issue}")
        
        print()
    
    def _print_final_summary(self):
        """Print the final validation summary."""
        
        total = self.results['total_files']
        success = self.results['successful_parses']
        fallback = self.results['fallback_parses']
        failed = self.results['failed_parses']
        
        print("=== VALIDATION SUMMARY ===")
        print(f"Total files processed: {total}")
        print(f"✅ Successful parses: {success} ({success/total*100:.1f}%)")
        print(f"🟡 Fallback parses: {fallback} ({fallback/total*100:.1f}%)")
        print(f"❌ Failed parses: {failed} ({failed/total*100:.1f}%)")
        print()
        
        # Calculate parser effectiveness
        effective_parses = success + fallback
        print(f"Parser effectiveness: {effective_parses}/{total} ({effective_parses/total*100:.1f}%)")
        
        # Show most common issues
        all_issues = []
        for result in self.results['detailed_results']:
            all_issues.extend(result['json_issues'])
        
        if all_issues:
            print("\n=== MOST COMMON ISSUES ===")
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            for issue, count in sorted_issues[:5]:
                print(f"{count:3d}x {issue}")
    
    def generate_detailed_report(self, output_file: str = "log_validation_report.json"):
        """Generate a detailed JSON report of all validation results."""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """Get a list of files that had parsing issues."""
        
        problematic = []
        for result in self.results['detailed_results']:
            if result['status'] in ['error', 'fallback'] or result['json_issues']:
                problematic.append(result)
        
        return problematic
    
    def test_parser_improvements(self):
        """Test specific improvements to the parser with problematic files."""
        
        print("\n=== TESTING PARSER IMPROVEMENTS ===")
        
        problematic_files = self.get_problematic_files()
        
        if not problematic_files:
            print("No problematic files found - parser is working perfectly!")
            return
        
        print(f"Testing improvements on {len(problematic_files)} problematic files...")
        
        improvements = 0
        for result in problematic_files[:5]:  # Test first 5 problematic files
            filename = result['filename']
            print(f"\nTesting: {filename}")
            
            # Read the original file
            log_file = self.logs_dir / filename
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Test with ultimate JSON fix
            try:
                fixed_content = self.parser._ultimate_json_fix(content)
                parsed = json.loads(fixed_content)
                print(f"  ✅ Ultimate fix successful!")
                improvements += 1
            except Exception as e:
                print(f"  ❌ Ultimate fix failed: {e}")
        
        print(f"\nImprovement rate: {improvements}/{len(problematic_files[:5])} files fixed")


def main():
    """Main function to run the log validation."""
    
    # Check if logs directory exists
    if not os.path.exists("logs"):
        print("❌ Logs directory not found!")
        return
    
    # Create validator and run validation
    validator = LogValidator()
    results = validator.validate_all_logs()
    
    # Generate detailed report
    validator.generate_detailed_report()
    
    # Test parser improvements on problematic files
    validator.test_parser_improvements()
    
    return results


if __name__ == "__main__":
    main()
