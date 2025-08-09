# Acknowledgments

This project builds upon and integrates with several outstanding open-source projects and models. We gratefully acknowledge the following contributors to the AI and open-source communities:

## OpenVINO GenAI
- **Source**: Intel Corporation
- **License**: Apache License 2.0
- **Repository**: https://github.com/openvinotoolkit/openvino.genai
- **Website**: https://docs.openvino.ai/
- **Usage**: This project uses configuration patterns, API examples, and best practices derived from OpenVINO GenAI official samples and documentation
- **Specific Attribution**: 
  - Pipeline initialization patterns from `samples/python/` directory
  - Stateful chat API usage following `chat_sample.py` patterns
  - Generation configuration examples and parameter handling
  - NPU optimization techniques and NPUW configuration patterns

## Qwen Models
- **Source**: Alibaba Cloud / Qwen Team
- **License**: Apache License 2.0 (with additional terms for model usage)
- **Repository**: https://github.com/QwenLM/Qwen
- **Website**: https://qwenlm.github.io/
- **Usage**: This project is designed to work with Qwen3-8B models and incorporates Qwen-specific optimizations
- **Specific Attribution**:
  - Special token definitions and handling for Qwen3 architecture
  - Chat template formats following Qwen3 specifications
  - Model-specific configuration parameters and constraints
  - Token filtering patterns for Qwen3's 26+ special tokens
- **Note**: Users must obtain Qwen3 models through official channels and comply with model license terms

## Gradio
- **Source**: HuggingFace Inc.
- **License**: Apache License 2.0
- **Repository**: https://github.com/gradio-app/gradio
- **Website**: https://gradio.app/
- **Usage**: User interface patterns and streaming implementations inspired by official Gradio examples
- **Specific Attribution**:
  - ChatInterface patterns and streaming response handling
  - Component layout and theming approaches
  - Event handling patterns for chat applications
  - Performance dashboard and metrics display techniques

## Transformers Library
- **Source**: HuggingFace Inc.
- **License**: Apache License 2.0
- **Repository**: https://github.com/huggingface/transformers
- **Website**: https://huggingface.co/transformers/
- **Usage**: Tokenizer integration and text processing utilities
- **Specific Attribution**:
  - AutoTokenizer usage patterns and configuration
  - Token handling and decoding techniques
  - Integration patterns with custom model pipelines

## OpenVINO Toolkit
- **Source**: Intel Corporation
- **License**: Apache License 2.0
- **Repository**: https://github.com/openvinotoolkit/openvino
- **Website**: https://openvino.ai/
- **Usage**: Core inference engine and optimization techniques
- **Specific Attribution**:
  - Property configuration patterns and device handling
  - Performance optimization techniques for Intel hardware
  - NPU-specific configuration and deployment patterns

## Python Dependencies
This project also uses several Python libraries, each with their respective licenses:
- **numpy**: BSD License
- **queue, threading, time**: Python Standard Library (PSF License)
- **dataclasses, typing**: Python Standard Library (PSF License)
- **pathlib, os, sys**: Python Standard Library (PSF License)

## Documentation and Learning Resources
We also acknowledge the broader community resources that influenced the development of this project:
- OpenVINO GenAI documentation and tutorials
- Intel Developer Zone articles on NPU optimization
- HuggingFace model hub documentation
- Gradio community examples and tutorials
- GitHub community discussions and issue threads

## Community Contributions
Special thanks to the open-source AI community for:
- Sharing optimization techniques and best practices
- Providing detailed documentation and examples
- Contributing to forums and discussions that helped solve implementation challenges
- Creating comprehensive guides for AI model deployment

---

## How to Cite

If you use this project in your research or applications, please consider citing the original sources:

### OpenVINO GenAI
```
@software{openvino_genai,
  title={OpenVINO GenAI},
  author={Intel Corporation},
  url={https://github.com/openvinotoolkit/openvino.genai},
  year={2024}
}
```

### Qwen Models
```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and others},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

### Gradio
```
@software{gradio,
  title={Gradio: Hassle-free sharing and testing of ML models in the wild},
  author={Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  url={https://gradio.app/},
  year={2019}
}
```

---

*This project is a community contribution and is not officially affiliated with Intel, Alibaba Cloud, HuggingFace, or any of the above-mentioned organizations.*