#!/usr/bin/env python3
"""
Enhanced Qwen3 OpenVINO GenAI Chat Application
==============================================

Main entry point for the modular, production-ready implementation of Qwen3-8B 
chat interface using OpenVINO GenAI with Intel NPU optimization and RAG capabilities.

Copyright (c) 2025 sbran
Licensed under the MIT License - see LICENSE file for details

Usage:
    python main.py                          # Use defaults from config.json
    python main.py --device CPU             # Force CPU device
    python main.py --model-path /path/model # Use custom model path
    python main.py --help                   # Show all options

Features:
- Complete Qwen3 NPUW optimization
- Dynamic system prompt configuration
- RAG document processing
- Professional performance monitoring  
- Robust error handling and diagnostics
- Modular architecture for maintainability
"""

import argparse
import os
import sys
from typing import Optional

# Import application modules
from app.config import initialize_config
from app.model import initialize_system_with_validation
from app.chat import initialize_globals as init_chat_globals
from app.ui import create_enhanced_interface, initialize_ui_globals
from app.streamer import streaming_metrics


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser with comprehensive options.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Qwen3 OpenVINO GenAI Chat Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use config.json defaults
  python main.py --device CPU                 # Force CPU device  
  python main.py --npu-profile conservative  # Use conservative NPU settings
  python main.py --model-path ./models/qwen3 # Custom model location
  python main.py --share                     # Enable public sharing (use with caution)
  python main.py --port 8080                 # Use custom port

Configuration Priority:
  1. Command-line arguments (highest priority)
  2. Environment variables
  3. config.json file
  4. Built-in defaults (lowest priority)

Environment Variables:
  QWEN3_MODEL_PATH     - Model directory path
  TARGET_DEVICE        - Target device (NPU, CPU, GPU, AUTO)
  NPU_PROFILE          - NPU optimization profile
  CACHE_DIR            - Cache directory location
  MAX_MESSAGE_LENGTH   - Maximum message length
  GENERATION_TIMEOUT   - Generation timeout in seconds
  GRADIO_SHARE         - Enable public sharing (true/false)
        """
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Path to the OpenVINO model directory (overrides config.json and env var)"
    )
    
    # Device configuration
    device_group = parser.add_argument_group("Device Configuration")
    device_group.add_argument(
        "--device",
        type=str,
        choices=["NPU", "CPU", "GPU", "AUTO"],
        help="Target device for inference (default: NPU)"
    )
    device_group.add_argument(
        "--npu-profile",
        type=str,
        choices=["conservative", "balanced", "aggressive"],
        help="NPU optimization profile (default: balanced)"
    )
    device_group.add_argument(
        "--cache-dir",
        type=str,
        help="OpenVINO cache directory path"
    )
    
    # Application configuration
    app_group = parser.add_argument_group("Application Configuration")
    app_group.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    app_group.add_argument(
        "--max-message-length",
        type=int,
        help="Maximum message length in characters"
    )
    app_group.add_argument(
        "--generation-timeout",
        type=float,
        help="Generation timeout in seconds"
    )
    
    # Gradio interface configuration
    gradio_group = parser.add_argument_group("Interface Configuration")
    gradio_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the interface to (default: 127.0.0.1)"
    )
    gradio_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the interface to (default: 7860)"
    )
    gradio_group.add_argument(
        "--share",
        action="store_true",
        help="Enable public sharing via Gradio (WARNING: Security risk)"
    )
    gradio_group.add_argument(
        "--auth",
        type=str,
        help="Basic authentication (format: username:password)"
    )
    gradio_group.add_argument(
        "--max-file-size",
        type=str,
        default="10mb",
        help="Maximum file upload size (default: 10mb)"
    )
    
    # Development options
    dev_group = parser.add_argument_group("Development Options")
    dev_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    dev_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate system requirements, don't start interface"
    )
    dev_group.add_argument(
        "--reset-metrics",
        action="store_true",
        help="Reset performance metrics on startup"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate model path if provided
    if args.model_path and not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Validate cache directory if provided
    if args.cache_dir:
        cache_parent = os.path.dirname(args.cache_dir)
        if cache_parent and not os.path.exists(cache_parent):
            try:
                os.makedirs(cache_parent, exist_ok=True)
            except (PermissionError, OSError) as e:
                print(f"❌ Error: Cannot create cache directory parent: {cache_parent} ({e})")
                sys.exit(1)
    
    # Validate authentication format
    if args.auth and ':' not in args.auth:
        print("❌ Error: Authentication must be in format 'username:password'")
        sys.exit(1)
    
    # Validate port range
    if not (1024 <= args.port <= 65535):
        print(f"❌ Error: Port must be between 1024 and 65535, got {args.port}")
        sys.exit(1)
    
    # Warn about security risks
    if args.share:
        print("⚠️ WARNING: Public sharing enabled. Your application will be accessible from the internet.")
        print("   Ensure you trust all users who might access it.")
    
    if args.host != "127.0.0.1":
        print(f"⚠️ WARNING: Binding to {args.host}. Make sure your firewall is properly configured.")


def setup_launch_config(args: argparse.Namespace) -> dict:
    """
    Setup Gradio launch configuration from arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with launch configuration
    """
    launch_config = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share or os.getenv("GRADIO_SHARE", "").lower() in ('true', '1', 'yes'),
        "show_error": True,
        "show_tips": True,
        "quiet": not args.debug,
        "max_file_size": args.max_file_size,
        "allowed_paths": []  # No file access by default for security
    }
    
    # Add authentication if provided
    if args.auth:
        username, password = args.auth.split(':', 1)
        launch_config["auth"] = (username, password)
        print(f"✅ Basic authentication enabled for user: {username}")
    
    return launch_config


def main():
    """Main application entry point"""
    
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup debug logging if requested
    if args.debug:
        print("🔧 Debug mode enabled")
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print("🚀 Starting Enhanced Qwen3 Chat Application")
    print("=" * 60)
    
    try:
        # Initialize configuration system
        print("🔧 Loading configuration...")
        config = initialize_config(args.config, args)
        
        if args.validate_only:
            print("🔍 Validation-only mode - checking system requirements...")
            # Import validation function
            from app.model import validate_system_requirements
            issues = validate_system_requirements()
            
            if issues:
                print("❌ Validation failed:")
                for i, issue in enumerate(issues, 1):
                    print(f"   {i}. {issue}")
                sys.exit(1)
            else:
                print("✅ All system requirements validated successfully")
                sys.exit(0)
        
        # Reset metrics if requested
        if args.reset_metrics:
            streaming_metrics.reset()
            print("📊 Performance metrics reset")
        
        # Initialize system with validation
        print("🚀 Initializing system...")
        pipeline, tokenizer, device_used, config_used, load_time = initialize_system_with_validation()
        
        # Initialize global instances
        init_chat_globals(pipeline, tokenizer)
        initialize_ui_globals(pipeline, device_used, config_used, load_time)
        
        # Create Gradio interface
        print("🌐 Creating Gradio interface...")
        demo = create_enhanced_interface()
        
        # Setup launch configuration
        launch_config = setup_launch_config(args)
        
        # Display startup information
        print("✨ Enhanced Qwen3 Chat System Ready!")
        print("=" * 60)
        print("Features Enabled:")
        print("   🎯 Dynamic System Prompts")
        print("   📚 RAG Document Processing") 
        print("   🔍 Advanced Token Filtering")
        print("   📊 Real-time Performance Monitoring")
        print("   🛡️ Security & Input Validation")
        print("   🏗️ Modular Architecture")
        print("=" * 60)
        print(f"🌐 Starting server on {args.host}:{args.port}")
        print(f"🎯 Device: {device_used} | Config: {config_used}")
        print(f"⏱️ Load Time: {load_time:.1f}s")
        
        if launch_config["share"]:
            print("🔗 Public sharing enabled - link will be displayed after startup")
        
        print("=" * 60)
        
        # Launch the application
        demo.queue(
            max_size=20,
            default_concurrency_limit=1  # NPU works best with single concurrent requests
        ).launch(**launch_config)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        sys.exit(0)
    
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print("   Use --debug flag for detailed error information")
        
        print("\n🆘 This may be a configuration or system issue")
        print("   Check your model path, device drivers, and dependencies")
        sys.exit(1)


if __name__ == "__main__":
    main()