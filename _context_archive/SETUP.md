# Setup Guide for Enhanced Qwen3 OpenVINO GenAI Chat

Quick setup guide for new users to get the application running.

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites Check
```bash
# Check Python version (3.8+ required)
python --version

# Check if you have git
git --version
```

### 2. Clone and Install
```bash
# Clone the repository
git clone <your-repo-url>
cd OpenVinoDev

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Install OpenVINO
Follow official installation guide: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html

**Quick install (most systems):**
```bash
pip install openvino
```

### 4. Get Qwen3 Model
Download Qwen3-8B INT4 model from official sources:
- **HuggingFace**: https://huggingface.co/Qwen/Qwen2.5-8B-Instruct
- **ModelScope**: https://modelscope.cn/models/qwen/Qwen2.5-8B-Instruct

Convert to OpenVINO format using `optimum-intel` if needed.

### 5. Configure Environment
```bash
# Set model path (adjust to your location)
export QWEN3_MODEL_PATH="/path/to/your/qwen3-8b-int4-ov"

# Optional: Set cache directory
export CACHE_DIR="./cache"
```

### 6. Run the Application
```bash
python gradio_qwen_enhanced.py
```

Open http://127.0.0.1:7860 in your browser.

## 🔧 Detailed Setup

### For Intel NPU Users
1. **Install NPU drivers** from Intel's official site
2. **Verify NPU availability**:
   ```bash
   python -c "import openvino as ov; print(ov.Core().available_devices)"
   ```
3. **Set NPU as target device**:
   ```bash
   export TARGET_DEVICE="NPU"
   ```

### For CPU-Only Users
```bash
export TARGET_DEVICE="CPU"
```

### Troubleshooting
- **"Model not found"**: Check `QWEN3_MODEL_PATH` environment variable
- **"NPU not available"**: Install NPU drivers or use CPU fallback
- **Import errors**: Ensure all dependencies from `requirements.txt` are installed

## 📚 Next Steps
- Read [README.md](README.md) for detailed features
- Check [DISCLAIMER.md](DISCLAIMER.md) for usage guidelines  
- See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for third-party attributions

## 🆘 Getting Help
- Check existing GitHub issues
- Review troubleshooting section in README.md
- Ensure all dependencies are correctly installed