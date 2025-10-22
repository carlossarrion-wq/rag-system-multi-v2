"""
Ultimate JSON Fixer for LLM Responses
Handles all common JSON issues including unescaped quotes within string values
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def ultimate_json_fix(json_str: str) -> str:
    """
    Ultimate JSON fixer that handles all common LLM JSON issues:
    1. Typographic quotes (curly quotes)
    2. Unescaped quotes in string values (main issue)
    3. Unescaped newlines and control characters
    4. Incomplete JSON structures
    5. Trailing commas
    
    Args:
        json_str: Raw JSON string from LLM that may have multiple issues
        
    Returns:
        Fixed JSON string that should be parseable
    """
    
    # Step 1: Try to parse as-is first
    try:
        json.loads(json_str)
        return json_str  # Already valid
    except json.JSONDecodeError:
        pass
    
    logger.debug("Starting ultimate JSON fix process")
    
    # Step 2: Fix typographic quotes first
    fixed = fix_typographic_quotes(json_str)
    
    # Step 3: Fix structural issues
    fixed = fix_structural_issues(fixed)
    
    # Step 4: Fix unescaped quotes in string values (MAIN ISSUE)
    fixed = fix_unescaped_quotes_smart(fixed)
    
    # Step 5: Fix control characters
    fixed = fix_control_characters(fixed)
    
    # Step 6: Final cleanup
    fixed = final_cleanup(fixed)
    
    return fixed


def fix_typographic_quotes(json_str: str) -> str:
    """Fix typographic quotes (curly quotes) that LLMs sometimes generate."""
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


def fix_unescaped_quotes_smart(json_str: str) -> str:
    """
    Smart fix for unescaped quotes within JSON string values.
    Uses a state machine approach to properly identify string boundaries.
    """
    
    result = []
    i = 0
    in_string = False
    escape_next = False
    string_start_pos = -1
    
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
                string_start_pos = i
                result.append(char)
                logger.debug(f"Starting string at position {i}")
            else:
                # We're in a string, check if this quote should end the string
                if is_string_terminator_smart(json_str, i):
                    # This quote ends the string
                    in_string = False
                    result.append(char)
                    logger.debug(f"Ending string at position {i} (started at {string_start_pos})")
                else:
                    # This quote is inside the string, escape it
                    result.append('\\"')
                    logger.debug(f"Escaped quote at position {i} (inside string started at {string_start_pos})")
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def is_string_terminator_smart(json_str: str, quote_pos: int) -> bool:
    """
    Smart determination if a quote at the given position terminates a string.
    Uses better heuristics to identify JSON structure.
    """
    
    # Look at what comes after the quote (ignoring whitespace)
    remaining = json_str[quote_pos + 1:].lstrip()
    
    if not remaining:
        return True  # End of string
    
    # Characters that typically follow a string value in JSON
    terminators = [',', '}', ']', ':']
    
    first_char = remaining[0]
    
    # If followed by a terminator, this is likely the end of the string
    if first_char in terminators:
        return True
    
    # Special case: if followed by another quote immediately, it's likely not a terminator
    # unless there's clear JSON structure after
    if first_char == '"':
        # Look further ahead to see if there's JSON structure
        next_remaining = remaining[1:].lstrip()
        if next_remaining and next_remaining[0] in terminators:
            return False  # This is likely a quote within the string
    
    # Additional heuristic: if followed by text that looks like it continues the sentence,
    # it's probably not a terminator
    if first_char.isalpha() or first_char in '.,;:!?':
        return False
    
    # If followed by whitespace and then a terminator, it's likely the end
    if remaining.lstrip() and remaining.lstrip()[0] in terminators:
        return True
    
    # Default to not terminating (safer to escape)
    return False


def fix_control_characters(json_str: str) -> str:
    """Fix unescaped control characters in string values."""
    
    # Enhanced control character fixing
    control_chars = {
        '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002',
        '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005',
        '\x06': '\\u0006', '\x07': '\\u0007', '\x08': '\\b',
        '\x09': '\\t', '\x0a': '\\n', '\x0b': '\\u000b',
        '\x0c': '\\f', '\x0d': '\\r', '\x0e': '\\u000e',
        '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011',
        '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014',
        '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017',
        '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a',
        '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d',
        '\x1e': '\\u001e', '\x1f': '\\u001f'
    }
    
    # Apply control character fixes
    for char, replacement in control_chars.items():
        if char in json_str:
            json_str = json_str.replace(char, replacement)
            logger.debug(f"Fixed control character: {repr(char)} -> {replacement}")
    
    return json_str


def final_cleanup(json_str: str) -> str:
    """Final cleanup and validation."""
    # Remove any duplicate escaping that might have been introduced
    fixed = json_str.replace('\\\\n', '\\n')
    fixed = fixed.replace('\\\\t', '\\t')
    fixed = fixed.replace('\\\\r', '\\r')
    
    # Fix any double-escaped quotes that aren't needed
    fixed = re.sub(r'\\\\(")', r'\1', fixed)
    
    return fixed


def test_ultimate_fixer():
    """Test the ultimate fixer with the user's problematic JSON."""
    
    print("=== ULTIMATE JSON FIXER TEST ===")
    print()
    
    # Test with the user's actual problematic JSON
    try:
        with open('user_problematic_json.json', 'r', encoding='utf-8') as f:
            problematic_json = f.read()
        
        print("Testing user's problematic JSON:")
        print(f"Original length: {len(problematic_json)} chars")
        
        # Test original
        try:
            json.loads(problematic_json)
            print("✅ Original JSON is valid")
        except json.JSONDecodeError as e:
            print(f"❌ Original JSON invalid: {e}")
            print(f"Error at position {e.pos}, line {e.lineno}, column {e.colno}")
            
            # Apply ultimate fix
            try:
                fixed = ultimate_json_fix(problematic_json)
                parsed = json.loads(fixed)
                print("✅ Ultimate fix successful!")
                print(f"Fixed length: {len(fixed)} chars")
                
                # Show key information
                if 'answer' in parsed:
                    print(f"Answer preview: {parsed['answer'][:100]}...")
                
                # Save the fixed version
                with open('user_json_ultimate_fixed.json', 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                print("✅ Saved fixed JSON to 'user_json_ultimate_fixed.json'")
                
                # Show what was fixed
                print("\n=== COMPARISON ===")
                print("Original problematic section:")
                print(repr(problematic_json[230:270]))
                print("Fixed section:")
                print(repr(fixed[230:270]))
                
                return True
                
            except Exception as e:
                print(f"❌ Ultimate fix failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
    except FileNotFoundError:
        print("❌ user_problematic_json.json not found")
        return False
    
    return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the test
    success = test_ultimate_fixer()
    
    if success:
        print("\n🎉 Successfully fixed the user's JSON!")
    else:
        print("\n❌ Failed to fix the user's JSON")
