"""
Enhanced JSON Cleaner for LLM Responses
Specifically handles typographic quotes and other common LLM JSON issues
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def fix_typographic_quotes(json_str: str) -> str:
    """
    Fix typographic quotes (curly quotes) that LLMs sometimes generate.
    
    Common problematic characters:
    - " (U+201C) LEFT DOUBLE QUOTATION MARK
    - " (U+201D) RIGHT DOUBLE QUOTATION MARK  
    - ' (U+2018) LEFT SINGLE QUOTATION MARK
    - ' (U+2019) RIGHT SINGLE QUOTATION MARK
    
    Args:
        json_str: JSON string with potential typographic quotes
        
    Returns:
        JSON string with straight quotes
    """
    # Map of typographic quotes to straight quotes
    quote_mapping = {
        '"': '"',  # LEFT DOUBLE QUOTATION MARK
        '"': '"',  # RIGHT DOUBLE QUOTATION MARK
        ''': "'",  # LEFT SINGLE QUOTATION MARK
        ''': "'",  # RIGHT SINGLE QUOTATION MARK
        '„': '"',  # DOUBLE LOW-9 QUOTATION MARK
        '‚': "'",  # SINGLE LOW-9 QUOTATION MARK
        '«': '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
        '»': '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
        '‹': "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
        '›': "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    }
    
    fixed_str = json_str
    replacements_made = 0
    
    for typographic, straight in quote_mapping.items():
        if typographic in fixed_str:
            count = fixed_str.count(typographic)
            fixed_str = fixed_str.replace(typographic, straight)
            replacements_made += count
            logger.debug(f"Replaced {count} instances of '{typographic}' with '{straight}'")
    
    if replacements_made > 0:
        logger.info(f"Fixed {replacements_made} typographic quote characters")
    
    return fixed_str


def enhanced_json_clean(json_str: str) -> str:
    """
    Enhanced JSON cleaner that handles all common LLM JSON issues:
    1. Typographic quotes (primary issue)
    2. Unescaped quotes in string values
    3. Unescaped newlines and control characters
    4. Incomplete JSON structures
    5. Trailing commas
    
    Args:
        json_str: Raw JSON string from LLM
        
    Returns:
        Fixed JSON string that should be parseable
    """
    
    # Step 1: Try to parse as-is first
    try:
        json.loads(json_str)
        return json_str  # Already valid
    except json.JSONDecodeError:
        pass
    
    logger.debug("Starting enhanced JSON cleaning process")
    
    # Step 2: Fix typographic quotes first (most common issue)
    fixed = fix_typographic_quotes(json_str)
    
    # Test if typographic quote fix alone solved the problem
    try:
        json.loads(fixed)
        logger.info("JSON fixed by correcting typographic quotes only")
        return fixed
    except json.JSONDecodeError:
        pass
    
    # Step 3: Fix structural issues
    fixed = fix_structural_issues(fixed)
    
    # Step 4: Fix unescaped quotes in string values
    fixed = fix_unescaped_quotes_in_strings(fixed)
    
    # Step 5: Fix control characters
    fixed = fix_control_characters(fixed)
    
    # Step 6: Final cleanup
    fixed = final_cleanup(fixed)
    
    return fixed


def fix_structural_issues(json_str: str) -> str:
    """Fix structural JSON issues like incomplete braces and trailing commas."""
    
    # Remove trailing whitespace and newlines
    fixed = json_str.rstrip()
    
    # Fix trailing comma at the end
    if fixed.endswith(','):
        fixed = fixed[:-1]
    
    # Count braces to determine if JSON is incomplete
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    
    if open_braces > close_braces:
        # Add missing closing braces
        missing_braces = open_braces - close_braces
        fixed += '}' * missing_braces
        logger.debug(f"Added {missing_braces} missing closing braces")
    
    # Handle incomplete arrays
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')
    
    if open_brackets > close_brackets:
        missing_brackets = open_brackets - close_brackets
        fixed += ']' * missing_brackets
        logger.debug(f"Added {missing_brackets} missing closing brackets")
    
    return fixed


def fix_unescaped_quotes_in_strings(json_str: str) -> str:
    """
    Fix unescaped quotes within JSON string values.
    This is complex as we need to identify string boundaries correctly.
    """
    
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(json_str):
        char = json_str[i]
        
        if escape_next:
            # This character is escaped, add as-is
            result.append(char)
            escape_next = False
        elif char == '\\':
            # This is an escape character
            result.append(char)
            escape_next = True
        elif char == '"':
            if not in_string:
                # Starting a new string
                in_string = True
                result.append(char)
            else:
                # We're in a string, check if this quote should end the string
                if is_string_terminator(json_str, i):
                    # This quote ends the string
                    in_string = False
                    result.append(char)
                else:
                    # This quote is inside the string, escape it
                    result.append('\\"')
                    logger.debug(f"Escaped quote at position {i}")
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def is_string_terminator(json_str: str, quote_pos: int) -> bool:
    """
    Determine if a quote at the given position terminates a string.
    Look at the context after the quote to make this determination.
    """
    
    # Look at what comes after the quote (ignoring whitespace)
    remaining = json_str[quote_pos + 1:].lstrip()
    
    if not remaining:
        return True  # End of string
    
    # Characters that typically follow a string value in JSON
    terminators = [',', '}', ']', ':']
    
    # Special case: if followed by another quote, it's likely not a terminator
    # unless there's a terminator after that quote
    if remaining.startswith('"'):
        # Look further ahead
        next_remaining = remaining[1:].lstrip()
        if next_remaining and next_remaining[0] in terminators:
            return False  # This is likely a quote within the string
    
    return remaining[0] in terminators


def fix_control_characters(json_str: str) -> str:
    """Fix unescaped control characters in string values."""
    
    # This is a simpler approach: replace common control characters
    # within string values only
    
    def fix_string_content(match):
        key = match.group(1)
        value = match.group(2)
        
        # Fix control characters
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        value = value.replace('\t', '\\t')
        value = value.replace('\b', '\\b')
        value = value.replace('\f', '\\f')
        
        return f'"{key}": "{value}"'
    
    # Pattern to match key-value pairs with string values
    # This is more targeted to avoid affecting JSON structure
    pattern = r'"([^"]+)":\s*"([^"]*(?:\\.[^"]*)*)"'
    
    try:
        fixed = re.sub(pattern, fix_string_content, json_str, flags=re.DOTALL)
        return fixed
    except Exception as e:
        logger.debug(f"Control character fix failed: {e}")
        return json_str


def final_cleanup(json_str: str) -> str:
    """Final cleanup and validation."""
    
    # Remove any duplicate escaping that might have been introduced
    fixed = json_str.replace('\\\\n', '\\n')
    fixed = fixed.replace('\\\\t', '\\t')
    fixed = fixed.replace('\\\\r', '\\r')
    
    # Fix any double-escaped quotes that aren't needed
    # Be careful not to break intentionally escaped quotes
    fixed = re.sub(r'\\\\(")', r'\1', fixed)
    
    return fixed


def test_enhanced_cleaner():
    """Test the enhanced cleaner with the user's problematic JSON."""
    
    print("=== ENHANCED JSON CLEANER TEST ===")
    print()
    
    # Test with the user's actual problematic JSON
    try:
        with open('user_problematic_json.json', 'r', encoding='utf-8') as f:
            problematic_json = f.read()
        
        print("Testing user's problematic JSON:")
        print(f"Original length: {len(problematic_json)} chars")
        
        # Show first few characters to identify the issue
        print(f"First 100 chars: {repr(problematic_json[:100])}")
        
        # Test original
        try:
            json.loads(problematic_json)
            print("✅ Original JSON is valid")
        except json.JSONDecodeError as e:
            print(f"❌ Original JSON invalid: {e}")
            print(f"Error at position {e.pos}, line {e.lineno}, column {e.colno}")
            
            # Apply enhanced fix
            try:
                fixed = enhanced_json_clean(problematic_json)
                parsed = json.loads(fixed)
                print("✅ Enhanced cleaning successful!")
                print(f"Fixed length: {len(fixed)} chars")
                
                # Show key information
                if 'answer' in parsed:
                    print(f"Answer preview: {parsed['answer'][:100]}...")
                
                # Save the fixed version
                with open('user_json_fixed.json', 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                print("✅ Saved fixed JSON to 'user_json_fixed.json'")
                
                return True
                
            except Exception as e:
                print(f"❌ Enhanced cleaning failed: {e}")
                return False
        
    except FileNotFoundError:
        print("❌ user_problematic_json.json not found")
        return False
    
    return False


def analyze_json_characters():
    """Analyze the characters in the problematic JSON to identify issues."""
    
    print("=== CHARACTER ANALYSIS ===")
    print()
    
    try:
        with open('user_problematic_json.json', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all quote-like characters
        quote_chars = set()
        for char in content:
            if char in '"\'""''„‚«»‹›':
                quote_chars.add(char)
        
        print("Quote-like characters found:")
        for char in sorted(quote_chars):
            count = content.count(char)
            print(f"  '{char}' (U+{ord(char):04X}): {count} times")
        
        # Show context around problematic positions
        print("\nContext around error position (char 234):")
        start = max(0, 234 - 50)
        end = min(len(content), 234 + 50)
        context = content[start:end]
        print(f"  ...{repr(context)}...")
        
        return quote_chars
        
    except FileNotFoundError:
        print("❌ user_problematic_json.json not found")
        return set()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Analyze characters first
    analyze_json_characters()
    print()
    
    # Run the test
    success = test_enhanced_cleaner()
    
    if success:
        print("\n🎉 Successfully fixed the user's JSON!")
    else:
        print("\n❌ Failed to fix the user's JSON")
