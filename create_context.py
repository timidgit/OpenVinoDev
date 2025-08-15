#!/usr/bin/env python3
"""
Python script to consolidate multiple project files into a single text file.
This is designed to create a comprehensive context file for an LLM.
"""

import os
from datetime import datetime
from pathlib import Path

def create_context_file():
    """Consolidate project files into a single context file for LLM consumption."""
    
    # Generate output filename with current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{current_time}_context_openvino.txt"
    
    # Define files to include with their correct paths
    files_to_include = [
        ("CLAUDE.md", "."),
        ("README.md", "."),
        ("requirements.txt", "."),
        ("main.py", "."),
        ("config.py", "app"),
        ("model.py", "app"), 
        ("config.json", "."),
        ("npu_patterns.py", "app"),
        ("streamer.py", "app"),
        ("chat.py", "app"),
        ("ui.py", "app"),
        ("__init__.py", "app"),
        ("agent.py", "app"),
        ("check_model_config.py", "."),
        ("export_model_for_npu.py", "."),
        ("test_chat_format.py", "."),
        ("test_minimal_gradio.py", "."), 
        ("test_simple_chat.py", "."),
        ("config.example.json", "_context_archive"),
        ("pytest.ini", "_context_archive"),
        ("test_config.py", "_context_archive/tests"),
        ("config.json", "phi3-128k-npu-fixed")
    ]
    
    # Remove existing output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑️  Removed existing {output_file}")
    
    print(f"📝 Creating context file: {output_file}")
    print("=" * 50)
    
    # Create the consolidated file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename, directory in files_to_include:
            file_path = Path(directory) / filename
            
            if file_path.exists():
                print(f"✅ Adding: {file_path}")
                
                # Write file header
                outfile.write(f"--- START OF {filename} ---\n\n")
                
                # Write file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                except UnicodeDecodeError:
                    # Handle binary or non-UTF8 files
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        content = infile.read()
                        outfile.write(content)
                
                # Write file footer
                outfile.write(f"\n\n--- END OF {filename} ---\n\n")
                
            else:
                print(f"❌ Missing: {file_path}")
                # Still add a placeholder in the output
                outfile.write(f"--- START OF {filename} ---\n\n")
                outfile.write(f"[FILE NOT FOUND: {file_path}]\n")
                outfile.write(f"\n--- END OF {filename} ---\n\n")
    
    print("=" * 50)
    print(f"✅ Consolidation complete!")
    print(f"📄 Output file: {output_file}")
    
    # Display file size
    file_size = os.path.getsize(output_file)
    if file_size > 1024 * 1024:
        print(f"📊 File size: {file_size / (1024 * 1024):.1f} MB")
    else:
        print(f"📊 File size: {file_size / 1024:.1f} KB")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = create_context_file()
        
        # Optional: Open the file for inspection
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--open":
            os.startfile(output_file)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")