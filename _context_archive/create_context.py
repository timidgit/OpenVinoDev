#!/usr/bin/env python3
"""
LLM Project Context Generator
============================

Python script to create a consolidated context file for the enhanced Phi-3 chat system.
This replaces the batch file functionality with cross-platform Python.
"""

import os
import glob
from pathlib import Path
from datetime import datetime


def should_exclude_path(path_str):
    """Check if a path should be excluded from context generation"""
    exclude_patterns = [
        '__pycache__', '.pytest_cache', '.git', 'node_modules', '.venv', 
        'cache', '.cache', 'combined_project', 'combined_llm_context',
        '.vs', '.vscode', 'bin', 'obj'
    ]
    
    path_lower = path_str.lower()
    return any(pattern in path_lower for pattern in exclude_patterns)


def get_project_files(project_root):
    """Get all relevant project files"""
    include_extensions = {
        '.py', '.json', '.txt', '.md', '.js', '.css', '.html', 
        '.bat', '.sh', '.yml', '.yaml', '.gitignore', '.dockerfile'
    }
    
    project_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(root, d))]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in include_extensions or file in ['.gitignore', 'Dockerfile']:
                if not should_exclude_path(file_path):
                    try:
                        # Check file size (skip very large files)
                        if os.path.getsize(file_path) < 1024 * 1024:  # 1MB limit
                            project_files.append(file_path)
                    except (OSError, IOError):
                        continue
    
    return sorted(project_files)


def create_context_file(project_root):
    """Create the combined context file"""
    output_file = os.path.join(project_root, 'combined_llm_context.txt')
    
    print(f"Creating LLM context file: {output_file}")
    print(f"Scanning project directory: {project_root}")
    
    project_files = get_project_files(project_root)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Write header
        out_file.write("=" * 80 + "\n")
        out_file.write("ENHANCED PHI-3 CHAT SYSTEM - COMPLETE PROJECT CONTEXT\n")
        out_file.write("=" * 80 + "\n")
        out_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_file.write(f"Project Root: {project_root}\n")
        out_file.write(f"Total Files: {len(project_files)}\n")
        out_file.write("=" * 80 + "\n\n")
        
        # Write summary of strategic roadmap implementation
        out_file.write("STRATEGIC ROADMAP IMPLEMENTATION STATUS:\n")
        out_file.write("✅ Phase 1: Foundational Refactoring & Consolidation\n")
        out_file.write("✅ Phase 2: Productionization & Scalability\n") 
        out_file.write("✅ Phase 3: Advanced AI Capabilities\n")
        out_file.write("   ✅ Phase 3.1: Advanced Document Parsing (unstructured)\n")
        out_file.write("   ✅ Phase 3.2: Cross-Encoder Reranking (BAAI/bge-reranker-base)\n")
        out_file.write("   ✅ Phase 3.3: Agentic Architecture (ReAct + function-calling)\n")
        out_file.write("\nThe application has been successfully transformed from prototype to\n")
        out_file.write("production-ready AI system with state-of-the-art capabilities.\n\n")
        out_file.write("=" * 80 + "\n\n")
        
        # Process each file
        included_count = 0
        skipped_count = 0
        
        for file_path in project_files:
            try:
                # Get relative path for cleaner output
                rel_path = os.path.relpath(file_path, project_root)
                
                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        print(f"Skipping binary file: {rel_path}")
                        skipped_count += 1
                        continue
                
                # Write file header
                out_file.write(f"FILE: {rel_path}\n")
                out_file.write("-" * 40 + "\n")
                out_file.write(content)
                out_file.write("\n\n" + "=" * 80 + "\n\n")
                
                included_count += 1
                if included_count % 10 == 0:
                    print(f"Processed {included_count} files...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped_count += 1
                continue
    
    print(f"\n✅ Context file created successfully!")
    print(f"📁 Output: {output_file}")
    print(f"📊 Included {included_count} files")
    print(f"⚠️ Skipped {skipped_count} files")
    
    return output_file


if __name__ == "__main__":
    # Get current directory as project root
    project_root = os.getcwd()
    create_context_file(project_root)