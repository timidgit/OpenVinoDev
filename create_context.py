#!/usr/bin/env python3
"""
Enhanced Context File Generator for External LLM Consumption
==========================================================

This script creates a comprehensive, structured context file optimized for external LLM analysis.
It includes project metadata, architecture diagrams, file categorization, and contextual information
to help external LLMs understand the codebase architecture and implementation patterns.

Features:
- Comprehensive project metadata and architecture overview
- Categorized file organization with purpose descriptions
- Code quality metrics and dependency analysis
- Configuration examples and deployment patterns
- Critical insights and debugging information
- Structured format optimized for LLM consumption

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class EnhancedContextGenerator:
    """Enhanced context file generator with comprehensive metadata and structuring."""
    
    def __init__(self):
        self.project_root = Path(".")
        self.output_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_enhanced_context_openvino.txt"
        
        # Categorized file structure for better LLM understanding
        self.file_categories = {
            "CORE_DOCUMENTATION": [
                ("CLAUDE.md", ".", "Primary development guidelines and architecture patterns"),
                ("README.md", ".", "User-facing documentation and installation guide"),
                ("requirements.txt", ".", "Core dependencies for basic functionality"),
                ("requirements-rag.txt", ".", "Optional dependencies for advanced RAG features")
            ],
            
            "APPLICATION_ENTRY_POINTS": [
                ("main.py", ".", "Primary application entry point with CLI interface"),
                ("config.json", ".", "Production configuration file")
            ],
            
            "CORE_MODULES": [
                ("__init__.py", "app", "Package initialization and public API"),
                ("config.py", "app", "Configuration management with 4-tier priority"),
                ("model.py", "app", "Pipeline deployment and system initialization"),
                ("chat.py", "app", "Core chat processing with RAG integration"),
                ("ui.py", "app", "Gradio interface creation and event handling"),
                ("streamer.py", "app", "Token streaming and filtering for LLMs")
            ],
            
            "SPECIALIZED_MODULES": [
                ("npu_patterns.py", "app", "NPU-specific optimization patterns for Phi-3"),
                ("agent.py", "app", "ReAct agent implementation with tool usage")
            ],
            
            "UTILITIES_AND_TOOLS": [
                ("check_model_config.py", ".", "Model configuration validation utility"),
                ("export_model_for_npu.py", ".", "NPU model export and optimization tool")
            ],
            
            "MODEL_CONFIGURATION": [
                ("config.json", "phi3-128k-npu-fixed", "Exported Phi-3 model configuration"),
                ("config.example.json", "_context_archive", "Configuration template with examples")
            ],
            
            "TEST_SUITE": [
                ("test_streaming_format.py", "tests", "Gradio streaming format compliance tests"),
                ("test_chat_format.py", ".", "Chat format validation tests"),
                ("test_minimal_gradio.py", ".", "Minimal Gradio interface tests"),
                ("test_simple_chat.py", ".", "Basic chat functionality tests"),
                ("pytest.ini", "_context_archive", "Test configuration and settings"),
                ("test_config.py", "_context_archive/tests", "Configuration testing patterns")
            ]
        }
        
        # Architecture patterns and critical insights
        self.architecture_insights = {
            "configuration_priority": "CLI → Environment → JSON → Defaults (4-tier system)",
            "openvino_api_pattern": "Stateful usage (start_chat/generate/finish_chat)",
            "npuw_hints": "FAST_COMPILE for prefill, BEST_PERF for generate (critical for NPU)",
            "gradio_format": "List[Dict[str, str]] with role/content keys (streaming compliance)",
            "device_fallback": "NPU → CPU automatic switching with appropriate configs",
            "context_limits": "NPU: 8k tokens (hardware), CPU: 128k tokens (full Phi-3)"
        }
    
    def generate_project_header(self) -> str:
        """Generate comprehensive project header with metadata."""
        header = f"""
{'='*80}
ENHANCED PHI-3 OPENVINO GENAI CHAT APPLICATION - CONTEXT FILE
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Purpose: Comprehensive context for external LLM analysis and development assistance
Project: Production-ready, modular Phi-3-mini-128k-instruct chat application

{'='*80}
PROJECT OVERVIEW
{'='*80}

ARCHITECTURE TYPE: Modular, production-ready implementation
PRIMARY MODEL: microsoft/Phi-3-mini-128k-instruct
OPTIMIZATION TARGET: Intel NPU with OpenVINO GenAI
FEATURES: RAG document processing, ReAct agents, real-time performance monitoring

KEY ARCHITECTURAL PRINCIPLES:
• 4-tier configuration priority (CLI > Env > JSON > Defaults)
• Stateful OpenVINO API usage (conversation state managed internally)
• Modular separation of concerns (config, model, chat, ui, streaming)
• Comprehensive error handling with device fallback (NPU → CPU)
• Security-focused input validation and sanitization

CRITICAL TECHNICAL CONSTRAINTS:
• NPU Context Limit: ~8,192 tokens (hardware limitation)
• Full 128k Context: Available on CPU device only
• NPUW Configuration: Must use FAST_COMPILE/BEST_PERF hints
• Gradio Compatibility: Requires List[Dict[str, str]] streaming format
• Legacy Naming: Some functions retain 'qwen3' names for backward compatibility

PRODUCTION QUALITY GATES:
✅ NPU compilation succeeds with target configuration
✅ CPU fallback operates correctly when NPU unavailable
✅ Gradio streaming works without format errors
✅ Performance metrics meet targets (>15 tok/sec NPU, >5 tok/sec CPU)
✅ Memory usage stays within device constraints
✅ Full 128k context utilization on CPU without crashes

{'='*80}
ARCHITECTURE DIAGRAM
{'='*80}

main.py (Entry Point)
├── app.config → ConfigurationLoader (4-tier priority system)
│   ├── CLI arguments (highest priority)
│   ├── Environment variables
│   ├── JSON configuration file
│   └── Built-in defaults (lowest priority)
│
├── app.model → deploy_llm_pipeline() (multi-tier NPU fallback)
│   ├── Enhanced Phi-3 patterns (npu_patterns.py)
│   ├── Manual NPUW configuration (fallback)
│   ├── Basic OpenVINO properties (minimal)
│   └── CPU fallback (automatic device switching)
│
├── app.ui → create_enhanced_interface() (Gradio with ChatInterface)
│   ├── Dynamic system prompts
│   ├── RAG document upload/processing
│   ├── Performance metrics dashboard
│   └── Real-time streaming responses
│
└── app.chat → enhanced_llm_chat() (stateful OpenVINO + RAG)
    ├── app.streamer → EnhancedLLMStreamer (Phi-3 token filtering)
    ├── app.agent → ReAct pattern (optional tool usage)
    ├── DocumentRAGSystem (vector search + cross-encoder reranking)
    └── InputValidator (security-focused sanitization)

{'='*80}
CRITICAL ARCHITECTURE PATTERNS
{'='*80}

1. CONFIGURATION ARCHITECTURE (4-tier priority):
   Priority Order: {self.architecture_insights['configuration_priority']}
   
2. OPENVINO API USAGE:
   Pattern: {self.architecture_insights['openvino_api_pattern']}
   Critical: Never reconstruct full conversation history
   
3. NPUW CONFIGURATION:
   Hints: {self.architecture_insights['npuw_hints']}
   Warning: Using generic PERFORMANCE_HINT causes compilation errors
   
4. GRADIO STREAMING:
   Format: {self.architecture_insights['gradio_format']}
   Critical: Must yield from generator, not return generator object
   
5. DEVICE FALLBACK:
   Strategy: {self.architecture_insights['device_fallback']}
   Context Limits: {self.architecture_insights['context_limits']}

{'='*80}
RECENT CRITICAL FIXES & DEBUGGING INSIGHTS
{'='*80}

1. GRADIO STREAMING FORMAT ERROR (RESOLVED):
   Problem: "Data incompatible with messages format" errors
   Root Cause: UI event handler returning generator instead of yielding
   Solution: Changed return enhanced_llm_chat(...) to yield from enhanced_llm_chat(...)
   
2. NPUW CONFIGURATION DISCOVERY:
   Problem: NPU compilation failures with generic performance hints
   Root Cause: Generic PERFORMANCE_HINT conflicts with NPUW-specific hints
   Solution: Use only NPUW-specific hints (FAST_COMPILE, BEST_PERF)
   
3. DEPENDENCY MANAGEMENT OPTIMIZATION:
   Problem: Heavy dependencies punishing new users
   Solution: Separated core (requirements.txt) and optional (requirements-rag.txt)
   
4. LEGACY FUNCTION DEPRECATION:
   Issue: Qwen3 naming in Phi-3 codebase causing confusion
   Solution: Added deprecation wrappers with clear migration guidance

{'='*80}
FILE CATEGORIES AND STRUCTURE
{'='*80}
"""
        return header
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Generate project statistics for better context."""
        stats = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "largest_files": [],
            "missing_files": []
        }
        
        for category, files in self.file_categories.items():
            for filename, directory, _ in files:
                file_path = Path(directory) / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    stats["total_files"] += 1
                    stats["total_size"] += file_size
                    
                    # Track file types
                    ext = file_path.suffix or "no_extension"
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
                    # Track largest files
                    stats["largest_files"].append((str(file_path), file_size))
                else:
                    stats["missing_files"].append(str(file_path))
        
        # Sort largest files
        stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = stats["largest_files"][:5]
        
        return stats
    
    def write_category_section(self, outfile, category: str, files: List[Tuple[str, str, str]]) -> None:
        """Write a categorized section with metadata."""
        outfile.write(f"\n{'='*80}\n")
        outfile.write(f"CATEGORY: {category.replace('_', ' ')}\n")
        outfile.write(f"{'='*80}\n\n")
        
        # Write category description
        category_descriptions = {
            "CORE_DOCUMENTATION": "Essential documentation for understanding the project architecture and usage",
            "APPLICATION_ENTRY_POINTS": "Main application entry points and configuration",
            "CORE_MODULES": "Core application modules implementing the main functionality",
            "SPECIALIZED_MODULES": "Specialized modules for NPU optimization and agent capabilities",
            "UTILITIES_AND_TOOLS": "Utility scripts and development tools",
            "MODEL_CONFIGURATION": "Model-specific configuration files and examples",
            "TEST_SUITE": "Test files and testing configuration"
        }
        
        if category in category_descriptions:
            outfile.write(f"DESCRIPTION: {category_descriptions[category]}\n\n")
        
        # Process each file in the category
        for filename, directory, description in files:
            file_path = Path(directory) / filename
            
            outfile.write(f"{'─'*60}\n")
            outfile.write(f"FILE: {filename}\n")
            outfile.write(f"PATH: {file_path}\n")
            outfile.write(f"PURPOSE: {description}\n")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                outfile.write(f"SIZE: {file_size:,} bytes\n")
                outfile.write(f"STATUS: ✅ Available\n")
                outfile.write(f"{'─'*60}\n\n")
                
                # Write file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        content = infile.read()
                        outfile.write(content)
                except Exception as e:
                    outfile.write(f"[ERROR READING FILE: {e}]\n")
                
                outfile.write(f"\n{'─'*60}\n")
                outfile.write(f"END OF FILE: {filename}\n")
                outfile.write(f"{'─'*60}\n\n")
                
            else:
                outfile.write(f"STATUS: ❌ Missing\n")
                outfile.write(f"{'─'*60}\n\n")
                outfile.write(f"[FILE NOT FOUND: {file_path}]\n\n")
    
    def generate_footer(self, stats: Dict[str, Any]) -> str:
        """Generate comprehensive footer with statistics."""
        footer = f"""
{'='*80}
CONTEXT FILE GENERATION SUMMARY
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Files Processed: {stats['total_files']}
Total Content Size: {stats['total_size']:,} bytes ({stats['total_size']/1024:.1f} KB)

FILE TYPE DISTRIBUTION:
"""
        for ext, count in sorted(stats["file_types"].items()):
            footer += f"  {ext}: {count} files\n"
        
        footer += f"\nLARGEST FILES:\n"
        for file_path, size in stats["largest_files"]:
            footer += f"  {file_path}: {size:,} bytes\n"
        
        if stats["missing_files"]:
            footer += f"\nMISSING FILES:\n"
            for missing_file in stats["missing_files"]:
                footer += f"  ❌ {missing_file}\n"
        
        footer += f"""
{'='*80}
USAGE INSTRUCTIONS FOR EXTERNAL LLM
{'='*80}

This context file is optimized for external LLM analysis. Key points:

1. ARCHITECTURE UNDERSTANDING:
   - Review the architecture diagram and critical patterns section
   - Understand the 4-tier configuration priority system
   - Note the stateful OpenVINO API usage pattern

2. CRITICAL CONSTRAINTS:
   - NPU context limit: ~8k tokens (hardware constraint)
   - NPUW hints must use FAST_COMPILE/BEST_PERF (not generic hints)
   - Gradio streaming requires List[Dict[str, str]] format

3. DEBUGGING INSIGHTS:
   - Recent fixes section contains solutions to common issues
   - Legacy naming (qwen3) is intentionally preserved for compatibility
   - Device fallback (NPU → CPU) is automatic and critical

4. DEVELOPMENT PATTERNS:
   - Follow modular architecture with clear separation of concerns
   - Use configuration management for all settings
   - Implement comprehensive error handling with user-friendly messages
   - Maintain backward compatibility while guiding toward modern APIs

5. TESTING AND VALIDATION:
   - Use the test suite to validate streaming format compliance
   - Validate NPU compilation before CPU fallback
   - Test both short (NPU) and long (CPU) context scenarios

{'='*80}
END OF ENHANCED CONTEXT FILE
{'='*80}
"""
        return footer
    
    def create_enhanced_context_file(self) -> str:
        """Create the enhanced context file with comprehensive structure."""
        print(f"🚀 Creating enhanced context file: {self.output_file}")
        print("=" * 60)
        
        # Remove existing file if it exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            print(f"🗑️  Removed existing {self.output_file}")
        
        # Get project statistics
        stats = self.get_file_statistics()
        
        # Create the consolidated file
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            # Write comprehensive header
            outfile.write(self.generate_project_header())
            
            # Write each categorized section
            for category, files in self.file_categories.items():
                print(f"📁 Processing category: {category}")
                self.write_category_section(outfile, category, files)
            
            # Write comprehensive footer
            outfile.write(self.generate_footer(stats))
        
        # Display completion information
        file_size = os.path.getsize(self.output_file)
        print("=" * 60)
        print("✅ Enhanced context generation complete!")
        print(f"📄 Output file: {self.output_file}")
        print(f"📊 File size: {file_size / (1024 * 1024):.1f} MB" if file_size > 1024 * 1024 else f"📊 File size: {file_size / 1024:.1f} KB")
        print(f"📈 Files processed: {stats['total_files']}")
        print(f"🎯 Optimized for external LLM consumption")
        
        return self.output_file


def main():
    """Main function to create enhanced context file."""
    try:
        generator = EnhancedContextGenerator()
        output_file = generator.create_enhanced_context_file()
        
        # Optional: Open the file for inspection
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--open":
            if os.name == 'nt':  # Windows
                os.startfile(output_file)
            else:  # Unix-like systems
                os.system(f"open '{output_file}'" if sys.platform == "darwin" else f"xdg-open '{output_file}'")
                
    except Exception as e:
        print(f"❌ Error creating enhanced context file: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()