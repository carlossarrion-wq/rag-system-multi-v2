"""
Test the real JSON file with enhanced debugging
"""

import json
import re

def fix_unescaped_quotes_robust(json_str: str) -> str:
    """
    More robust version that handles multiline JSON with unescaped quotes.
    """
    # First, let's try a simpler approach using regex
    # Find patterns like: "text with "quoted" content"
    # And replace with: "text with \"quoted\" content"
    
    def replace_unescaped_quotes(match):
        full_match = match.group(0)
        key = match.group(1)
        value = match.group(2)
        
        # Escape quotes within the value, but not the delimiting quotes
        # Split by quotes and escape every other one (the ones inside)
        parts = value.split('"')
        if len(parts) > 1:
            # Escape quotes that are not at the beginning or end
            escaped_parts = [parts[0]]  # First part (before any quote)
            for i in range(1, len(parts)):
                if i == len(parts) - 1 and parts[i] == '':
                    # Last empty part after final quote - this means the string ended with a quote
                    escaped_parts.append('\\""')
                    break
                else:
                    escaped_parts.append('\\"' + parts[i])
            
            escaped_value = ''.join(escaped_parts)
        else:
            escaped_value = value
        
        return f'"{key}": "{escaped_value}"'
    
    # Pattern to match JSON key-value pairs where value contains unescaped quotes
    # This is more targeted than the character-by-character approach
    pattern = r'"([^"]+)":\s*"([^"]*"[^"]*(?:"[^"]*)*)"'
    
    result = re.sub(pattern, replace_unescaped_quotes, json_str, flags=re.DOTALL)
    
    return result


def test_with_real_file():
    """Test with the actual problematic JSON file."""
    
    # Read the file
    with open('json_to_validate.json', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Original JSON (first 200 chars):")
    print(repr(content[:200]))
    print()
    
    # Try original parsing
    try:
        json.loads(content)
        print("✅ Original JSON is valid")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Original JSON invalid: {e}")
        print(f"Error at position: {e.pos}")
        print()
    
    # Try the robust fix
    try:
        print("Applying robust unescaped quotes fix...")
        fixed = fix_unescaped_quotes_robust(content)
        
        print("Fixed JSON (first 200 chars):")
        print(repr(fixed[:200]))
        print()
        
        # Try parsing the fixed version
        parsed = json.loads(fixed)
        print("✅ Robust fix successful!")
        print(f"Parsed keys: {list(parsed.keys())}")
        
        # Save the fixed version
        with open('json_to_validate_fixed.json', 'w', encoding='utf-8') as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print("✅ Saved fixed JSON to json_to_validate_fixed.json")
        
    except Exception as e:
        print(f"❌ Robust fix failed: {e}")
        
        # Let's try a manual approach for this specific case
        print("\nTrying manual fix for specific case...")
        manual_fixed = content.replace('"Gestión de proveedores"', '\\"Gestión de proveedores\\"')
        
        try:
            parsed = json.loads(manual_fixed)
            print("✅ Manual fix successful!")
            
            with open('json_to_validate_fixed.json', 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print("✅ Saved manually fixed JSON")
            
        except Exception as e2:
            print(f"❌ Manual fix also failed: {e2}")


if __name__ == "__main__":
    test_with_real_file()
