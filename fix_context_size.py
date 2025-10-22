#!/usr/bin/env python3
"""
Script to fix the context size issue by implementing more aggressive context reduction
"""

import os
import sys

def create_context_reducer():
    """Create an enhanced context reducer for the document context enhancer"""
    
    context_reducer_code = '''
def reduce_context_for_images(self, context: str, max_tokens: int = 100000) -> str:
    """
    Reduce context size when images are present to stay within token limits
    
    Args:
        context: The original context string
        max_tokens: Maximum tokens to allow for context (default 100K to leave room for images)
    
    Returns:
        Reduced context string
    """
    # Rough estimation: 4 characters per token
    max_chars = max_tokens * 4
    
    if len(context) <= max_chars:
        return context
    
    # Split context into sections
    lines = context.split('\\n')
    
    # Priority order for keeping content:
    # 1. Headers and titles (lines starting with #, ##, etc.)
    # 2. Short lines (likely important summaries)
    # 3. Lines with keywords related to diagrams
    # 4. Other content
    
    priority_lines = []
    diagram_lines = []
    short_lines = []
    other_lines = []
    
    diagram_keywords = ['diagrama', 'flujo', 'proceso', 'usuario', '700', '900', 'nif', 'workflow']
    
    for line in lines:
        line_lower = line.lower()
        
        if line.startswith('#') or line.startswith('**') or line.startswith('Title:'):
            priority_lines.append(line)
        elif any(keyword in line_lower for keyword in diagram_keywords):
            diagram_lines.append(line)
        elif len(line.strip()) < 100 and line.strip():
            short_lines.append(line)
        else:
            other_lines.append(line)
    
    # Reconstruct context with priorities
    reduced_lines = []
    current_chars = 0
    
    # Add priority content first
    for line_group in [priority_lines, diagram_lines, short_lines, other_lines]:
        for line in line_group:
            if current_chars + len(line) + 1 < max_chars:
                reduced_lines.append(line)
                current_chars += len(line) + 1
            else:
                break
        if current_chars >= max_chars * 0.9:  # Stop at 90% to be safe
            break
    
    reduced_context = '\\n'.join(reduced_lines)
    
    # Add truncation notice
    if len(reduced_context) < len(context):
        reduced_context += "\\n\\n[Context truncated to fit within token limits when processing images]"
    
    return reduced_context
'''
    
    return context_reducer_code

def update_document_context_enhancer():
    """Update the document context enhancer to handle large contexts better"""
    
    file_path = "src/agent/document_context_enhancer.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📝 Reading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the method already exists
    if 'reduce_context_for_images' in content:
        print("✅ Context reducer method already exists")
        return True
    
    # Find the class definition
    class_start = content.find('class DocumentContextEnhancer:')
    if class_start == -1:
        print("❌ Could not find DocumentContextEnhancer class")
        return False
    
    # Find a good place to insert the new method (before the last method)
    insert_pos = content.rfind('def ', class_start)
    if insert_pos == -1:
        print("❌ Could not find insertion point")
        return False
    
    # Find the start of the line
    line_start = content.rfind('\n', 0, insert_pos)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    
    # Insert the new method
    context_reducer = create_context_reducer()
    indented_method = '\n'.join('    ' + line for line in context_reducer.split('\n'))
    
    new_content = (
        content[:line_start] + 
        indented_method + '\n\n    ' +
        content[line_start:]
    )
    
    # Write the updated content
    print(f"💾 Writing updated {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Document context enhancer updated with context reducer")
    return True

def update_llm_client_to_use_context_reducer():
    """Update the LLM client to use the context reducer when images are present"""
    
    file_path = "src/generation/llm_client_fixed.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📝 Reading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if context reduction is already implemented
    if 'reduce_context_for_images' in content:
        print("✅ Context reduction already implemented in LLM client")
        return True
    
    # Find the generate_with_citations method where context is built
    method_start = content.find('def generate_with_citations(')
    if method_start == -1:
        print("❌ Could not find generate_with_citations method")
        return False
    
    # Find the end of the method
    method_end = content.find('\n    def ', method_start)
    if method_end == -1:
        method_end = len(content)
    
    # Extract the method
    method_content = content[method_start:method_end]
    
    # Check if context reduction is already there
    if 'reduce_context_for_images' in method_content:
        print("✅ Context reduction already implemented")
        return True
    
    # Find where context is built (after context = "\n\n".join(context_parts))
    insert_point = method_content.find('context = "\\n\\n".join(context_parts)')
    if insert_point == -1:
        print("❌ Could not find context building point")
        return False
    
    # Find the end of that line
    line_end = method_content.find('\n', insert_point)
    if line_end == -1:
        line_end = len(method_content)
    
    # Insert context reduction after context is built
    reduction_code = '''
        
        # Reduce context size when images are present to avoid token limit
        if images_data and hasattr(self, 'document_enhancer') and hasattr(self.document_enhancer, 'reduce_context_for_images'):
            original_length = len(context)
            context = self.document_enhancer.reduce_context_for_images(context, max_tokens=100000)
            logger.info(f"Context reduced from {original_length} to {len(context)} characters for image processing")'''
    
    # Insert the reduction code
    new_method_content = (
        method_content[:line_end] +
        reduction_code +
        method_content[line_end:]
    )
    
    # Replace in the full content
    new_content = content[:method_start] + new_method_content + content[method_end:]
    
    # Write the updated content
    print(f"💾 Writing updated {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ LLM client updated to use context reducer")
    return True

def main():
    print("🔧 FIXING CONTEXT SIZE ISSUE")
    print("=" * 50)
    
    print("📊 Analysis from logs:")
    print("   • 3 images: 222,262 bytes total")
    print("   • Base64: 296,356 characters (~74K tokens)")
    print("   • Error: 'Input is too long' even without images")
    print("   • Issue: Text context + images exceed 180K token limit")
    print()
    
    print("🛠️  Implementing solution:")
    print("   1. Add context reducer to DocumentContextEnhancer")
    print("   2. Update LLM client to use context reducer when images present")
    print("   3. Limit context to 100K tokens when images are included")
    print()
    
    success = True
    
    # Update document context enhancer
    if not update_document_context_enhancer():
        success = False
    
    # Update LLM client
    if not update_llm_client_to_use_context_reducer():
        success = False
    
    if success:
        print("\n✅ CONTEXT SIZE FIX COMPLETED")
        print("📋 Changes made:")
        print("   • Added reduce_context_for_images method to DocumentContextEnhancer")
        print("   • Updated LLM client to reduce context when images are present")
        print("   • Context limited to 100K tokens (400K chars) when images included")
        print("   • Prioritizes diagram-related content and headers")
        print()
        print("🚀 Next steps:")
        print("   1. Deploy changes to EC2")
        print("   2. Test with 'diagrama del usuario 900' query")
        print("   3. Verify images are processed successfully")
    else:
        print("\n❌ SOME UPDATES FAILED")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
