

# **Strategic Roadmap for the Enhanced Phi-3 Chat System: From Prototype to Production-Grade AI Application**

### **Expert Persona**

**Systems Architect and MLOps Strategist:** With over 15 years of experience designing and deploying scalable, high-performance machine learning systems, this expert specializes in bridging the gap between cutting-edge AI research and robust, enterprise-ready production environments. Their expertise lies in MLOps, system architecture, performance optimization for specialized hardware like NPUs, and creating strategic roadmaps that balance immediate needs with long-term scalability and maintainability. They provide a critical, systems-level perspective focused on building resilient, efficient, and future-proof AI applications.

---

### **Executive Summary**

This report presents a comprehensive analysis and strategic development roadmap for the Enhanced Phi-3 Chat System. The project currently stands as a powerful demonstration of Intel NPU-accelerated inference with OpenVINO, featuring a sophisticated modular architecture and advanced capabilities such as Retrieval-Augmented Generation (RAG) and dynamic system prompts. Its strengths lie in its deep NPU-specific optimizations, professional user interface, and robust configuration management system.

However, the analysis reveals critical areas that require immediate attention to ensure long-term viability and production readiness. The project carries significant technical debt, most notably in the form of a lingering monolithic script, legacy naming conventions from its prior use of the Qwen3 model, and a complete absence of an automated testing framework. These issues, if left unaddressed, will impede development velocity, increase maintenance costs, and introduce configuration risks.

To address these challenges, this report outlines a three-phase strategic roadmap:

1. **Phase 1: Foundational Refactoring and Consolidation.** This phase focuses on eliminating technical debt by unifying the architecture, refactoring legacy naming, and establishing a CI/CD pipeline with automated testing. These are non-negotiable steps to create a stable and maintainable codebase.  
2. **Phase 2: Productionization and Scalability.** This phase prepares the application for reliable deployment by introducing Docker containerization and enhancing the user interface with interactive controls for generation parameters, moving it from a developer tool to a robust application.  
3. **Phase 3: Advanced Feature Expansion.** This phase elevates the application's core AI capabilities to state-of-the-art by evolving the RAG pipeline with advanced document parsing and reranking, and introducing an agentic architecture with function-calling capabilities.

Beyond this roadmap, the report explores long-term strategic opportunities, including model customization through fine-tuning and a migration to the OpenVINO™ Model Server (OVMS) for an enterprise-grade microservices architecture. Executing this strategy will transform the project from an advanced prototype into a scalable, production-ready, and future-proof AI application.

---

## **Part I: Current State Architectural Assessment**

This part provides a comprehensive and critical analysis of the project as it exists today. The goal is to establish a clear, evidence-based baseline from which the strategic roadmap in Part II can be built.

### **Section 1: Analysis of the Dual Architecture and Technical Debt**

The project's most significant structural challenge stems from its evolutionary development, resulting in architectural inconsistencies and technical debt that must be addressed to ensure future maintainability and stability.

#### **1.1 The Modular vs. Monolithic Dichotomy**

The repository currently contains two distinct and conflicting architectural patterns for launching the application. The modern approach is a modular architecture orchestrated by main.py, which leverages discrete components from the app/ directory for configuration, model deployment, streaming, chat logic, and UI rendering.1 This structure adheres to software engineering best practices by promoting a clear separation of concerns, which enhances readability, simplifies debugging, and makes the system easier to extend.

In stark contrast, the legacy gradio\_qwen\_enhanced.py script represents a monolithic architecture, combining all application logic into a single, large file.1 While potentially simpler for initial prototyping, this approach is inherently difficult to maintain and scale.1

The existence of these two parallel architectures is a source of significant friction. While developer guidance in CLAUDE.md correctly directs new development toward the modular app/ structure, user-facing documentation in README.md and SETUP.md still presents the monolithic script as a valid entry point.1 This ambiguity creates cognitive overhead for developers and users alike. Any feature enhancement or bug fix must be considered in the context of both architectures, effectively doubling the maintenance burden or, more likely, leading to the legacy script becoming dangerously out of sync. This architectural duality directly impedes development velocity and increases the risk of introducing regressions, making it an unsustainable model for a production-grade application.

#### **1.2 Legacy Naming Conventions Post-Migration**

The project has successfully migrated its core functionality to the microsoft/Phi-3-mini-128k-instruct model, a move confirmed by the config.json file and the main.py entry point.1 Despite this migration, numerous artifacts throughout the codebase retain names from the previous

Qwen3 model. This includes file paths (context/qwen3\_model\_context/), class names (EnhancedQwen3Streamer), function names (deploy\_qwen3\_pipeline, enhanced\_qwen3\_chat), and environment variables (QWEN3\_MODEL\_PATH).1

This is more than a cosmetic issue; it represents a form of technical debt that actively increases configuration risk. The NPU optimization patterns, for example, reside in a directory named qwen3\_model\_context but are being applied to a Phi-3 model.1 While these patterns may be compatible now, a future update to OpenVINO or NPU drivers could introduce model-specific optimizations that a developer might overlook, assuming the "Qwen3" optimizations are irrelevant to Phi-3. This could lead to silent performance degradation or outright compilation failures. The legacy naming conventions create a misleading narrative about the code's function, making correct maintenance and future upgrades more difficult and error-prone.

### **Section 2: NPU Optimization and Hardware Coupling**

A core strength of the project is its deep and effective optimization for Intel Neural Processing Units (NPUs), though this comes at the cost of configuration flexibility.

#### **2.1 Effective Use of NPUW Configuration**

The application demonstrates a sophisticated understanding of the requirements for deploying LLMs on Intel NPUs via OpenVINO. The code correctly implements NPU Wrapper (NPUW) hints, such as "NPU\_USE\_NPUW": "YES" and "NPUW\_LLM": "YES", and sets critical parameters like NPUW\_LLM\_MAX\_PROMPT\_LEN to manage the NPU's memory and performance characteristics.1 The inclusion of NPU profiles ("conservative," "balanced," "aggressive") provides users with valuable control to tune the trade-off between performance and resource consumption.1 This level of detailed hardware-specific configuration is a significant strength and is essential for achieving optimal performance on the target device.

#### **2.2 Configuration Rigidity and Lack of Abstraction**

While the NPU settings are effective, they are hardcoded within the get\_npu\_config methods in both the modular and monolithic codebases.1 These critical values are not exposed in

config.json or as command-line arguments, tightly coupling the application logic to a specific generation of hardware and driver. If a future NPU or an updated driver requires different hint values, a developer would need to modify the core application code rather than simply updating a configuration file.

This design prioritizes deep optimization for the current hardware target over long-term portability and maintainability. As the OpenVINO ecosystem rapidly evolves with support for new models and hardware 2, this tight coupling presents a maintenance challenge. A more robust MLOps approach would abstract these hardware-specific details into the configuration layer, transforming hardware adaptation from a development task into a more flexible deployment configuration task.

### **Section 3: Core Feature Implementation Review**

The application's key features, RAG and security validation, are well-implemented but show clear areas for future maturation.

#### **3.1 RAG System \- Foundational but Limited**

The DocumentRAGSystem provides a solid foundation for retrieval-augmented generation.1 It uses a standard and effective stack, including

langchain, RecursiveCharacterTextSplitter for chunking, a lightweight sentence-transformers/all-MiniLM-L6-v2 model for embeddings, and an in-memory FAISS vector store for fast retrieval. This setup is well-suited for processing the plain-text file types listed in the UI, such as .txt, .md, and .py.

However, the system's effectiveness is constrained by its simplistic "shallow parsing" strategy. The backend logic reads all supported files as raw text, discarding any inherent structure. For complex, semi-structured documents common in enterprise settings, such as PDFs with tables and columns or DOCX files with headers and sections, this approach is insufficient. It will fail to correctly interpret document layout, leading to garbled context being fed into the vector store and, consequently, irrelevant or nonsensical search results. This inability to deeply understand document structure is the single biggest bottleneck preventing the RAG feature from being truly production-grade.

#### **3.2 Security and Input Validation**

The inclusion of the InputValidator class is a commendable and critical feature that demonstrates a production-oriented mindset.1 The class implements methods to validate user messages against excessive length, suspicious patterns (e.g.,

\<script\> tags to prevent XSS), and unusual character distributions. It also includes a sanitization method to normalize whitespace and remove control characters. This proactive approach to security is often overlooked in prototype applications and represents a significant strength, mitigating common vectors for abuse and denial-of-service attacks.

### **Section 4: User Interface and Backend Integration**

The Gradio-based user interface is a standout feature, though its integration with the backend reveals a critical inconsistency between documentation and implementation.

#### **4.1 Professional UI Design**

The user interface, defined in app/ui.py, is a major strength. It leverages gr.Blocks to create a highly professional and usable front-end that goes far beyond a basic proof-of-concept.1 The use of custom CSS, collapsible accordions for advanced features like RAG and system prompt configuration, and a real-time performance monitoring panel provides a rich user experience. This design effectively exposes the application's advanced capabilities in an intuitive and accessible manner, aligning with best practices for building custom Gradio applications.5

#### **4.2 Critical Data Format Inconsistency**

A significant issue exists in how chat history is managed and documented. The developer guide, CLAUDE.md, explicitly warns against using a list of dictionaries (\[{"role": "user",...}\]) for chat history, stating that the correct format for gr.ChatInterface is a list of lists (\[\["user\_message", "bot\_response"\]\]).1 However, the legacy

gradio\_qwen\_enhanced.py script uses the dictionary-based format, while the modern app/chat.py correctly uses the list-of-lists format.1

This contradiction between the developer guide, the legacy code, and the modern implementation points to a flawed refactoring and documentation process. Outdated or incorrect documentation is often more damaging than no documentation, as it can actively mislead developers and waste significant time on debugging. This inconsistency highlights a need to treat documentation as a first-class citizen of the codebase, ensuring it is updated and validated as part of the standard development and review cycle.

---

## **Part II: Strategic Development Roadmap**

This part provides a phased, prioritized plan to address the issues identified in Part I, pay down technical debt, and mature the application into a production-ready system.

### **Section 5: Phase 1 \- Foundational Refactoring and Consolidation**

This initial phase is critical for establishing a stable and maintainable foundation. Its primary goal is to eliminate technical debt and introduce professional software development practices.

#### **5.1 Unifying the Architecture**

The first and most important action is to formally deprecate and remove the gradio\_qwen\_enhanced.py script. All development, testing, and documentation (including README.md and SETUP.md) must be standardized on the main.py entry point and the app/ modular architecture. This single action will resolve the architectural duality, eliminate code duplication, and halve the ongoing maintenance burden, creating a single source of truth for the application's logic.

#### **5.2 Model-Agnostic Refactoring**

To align the codebase with its current functionality and prepare it for future model integrations, a repository-wide refactoring of legacy "Qwen3" naming is required. Components should be renamed to be model-agnostic or specific to the current Phi-3 model. Key changes should include:

* EnhancedQwen3Streamer \-\> EnhancedLLMStreamer  
* deploy\_qwen3\_pipeline \-\> deploy\_llm\_pipeline  
* context/qwen3\_model\_context \-\> context/llm\_optimization\_patterns  
* QWEN3\_MODEL\_PATH (environment variable) \-\> MODEL\_PATH

This refactoring resolves the configuration risk identified previously and makes the codebase more intuitive. It paves the way for easier integration of other OpenVINO-supported models like Llama, Mistral, or Gemma in the future.7

#### **5.3 Implementing a Testing and CI Framework**

The absence of automated testing is a major production readiness gap. A formal testing suite and a Continuous Integration (CI) pipeline must be established.

* **Testing Suite:**  
  * **Linting:** Integrate flake8 to enforce a consistent code style and catch syntax errors.  
  * **Unit Tests:** Use pytest to create unit tests for core, non-UI logic. High-priority targets for initial tests include the InputValidator in app/chat.py and the ConfigurationLoader in app/config.py, as their logic is critical and easily testable.  
* **Continuous Integration:**  
  * Create a CI pipeline using GitHub Actions by adding a workflow file (e.g., .github/workflows/ci.yml). This workflow should automatically run the linter and the pytest suite on every push and pull request.

Implementing a CI pipeline represents a fundamental shift from slow, error-prone manual validation to an automated assurance process. It provides a quality gate that offers rapid feedback to developers, prevents regressions, and builds confidence in the codebase. This automation is the bedrock of modern, agile development and is essential for increasing development velocity safely.10

### **Section 6: Phase 2 \- Productionization and Scalability**

With a stable foundation, this phase focuses on packaging the application for reliable, repeatable deployment and enhancing its utility for end-users.

#### **6.1 Containerization with Docker**

A Dockerfile must be created to containerize the application, which is a critical step for production deployment. The Dockerfile will define a reproducible environment by:

1. Starting from an official OpenVINO development image, such as openvino/ubuntu22\_dev, which includes necessary libraries.13  
2. Ensuring all required system dependencies and NPU drivers are installed within the container.15  
3. Copying the requirements.txt file and installing all Python packages.  
4. Copying the application source code and configuration files.  
5. Exposing the Gradio port (e.g., 7860\) and configuring the server to listen on all network interfaces (0.0.0.0) to be accessible from outside the container.16  
6. Defining the CMD instruction to launch the application via python main.py.

Containerization provides a consistent, isolated environment that eliminates "it works on my machine" issues, ensuring the application runs identically across development, staging, and production environments. It also greatly simplifies deployment and scaling.16

#### **6.2 Continuous Deployment (CD) Pipeline**

The CI pipeline established in Phase 1 should be extended into a full Continuous Deployment (CD) pipeline. After the linting and testing stages pass on the main branch, a new job in the GitHub Actions workflow should automatically:

1. Build the Docker image using the Dockerfile.  
2. Tag the image with a version number or commit hash for traceability.  
3. Push the tagged image to a container registry, such as Docker Hub or GitHub Container Registry.

This automates the release process, ensuring that every change merged into the main branch produces a tested, versioned, and deployable artifact. This completes the CI/CD loop, a cornerstone of modern MLOps practices.

#### **6.3 Enhancing UI Interactivity**

To increase the application's utility, key LLM generation parameters should be exposed as interactive controls in the Gradio UI. An "Advanced Generation Settings" accordion can be added to house components like gr.Slider for temperature and top\_p, and gr.Number for max\_new\_tokens. These components would pass their values as additional inputs to the main chat function.

The current generation configuration is static, limiting experimentation. Allowing users to modify these parameters in real-time transforms the application from a simple chatbot into a powerful tool for prompt engineering and exploring the model's creative capabilities. Gradio's gr.Blocks architecture fully supports this dynamic interaction.6

### **Section 7: Phase 3 \- Advanced Feature Expansion**

This phase focuses on evolving the application's core AI capabilities to match the state-of-the-art in generative AI.

#### **7.1 Evolving the RAG Pipeline: Advanced Parsing**

To overcome the "Shallow Parsing" bottleneck, the RAG system's document ingestion logic must be upgraded. The basic open(file).read() approach should be replaced with a sophisticated, pipeline-based document parsing library capable of handling complex, structured formats like PDF and DOCX. This will enable the system to extract text while preserving layout, identifying tables, and handling figures, resulting in significantly higher-quality context for the LLM. The choice of library should be guided by factors such as format support, extraction quality, and ease of integration.

**Table 7.1: Comparison of Advanced Document Parsing Libraries**

| Library | Supported Formats | Table Extraction | Layout Awareness | Integration | Execution Model |
| :---- | :---- | :---- | :---- | :---- | :---- |
| unstructured | PDF, DOCX, PPTX, HTML, etc. 21 | Good (via different strategies) 21 | High (hi\_res strategy) 21 | langchain native 21 | Local / API 21 |
| LlamaParse | PDF, DOCX, PPTX, etc. 22 | Excellent (LLM-based) 23 | High 22 | LlamaIndex native 22 | API-first 23 |
| PyMuPDF | PDF 24 | Basic (text extraction only) 24 | Low (extracts raw text stream) 24 | Manual | Local 24 |
| Docling | PDF, DOCX, images, audio 25 | Advanced 25 | High 25 | langchain native 25 | Local / Actor 25 |

#### **7.2 Evolving the RAG Pipeline: Cross-Encoder Reranking**

The accuracy of the RAG pipeline can be further improved by adding a second-stage reranker. The current single-stage process relies on vector search (a bi-encoder approach), which is fast but can retrieve documents that are only superficially related to the query. A two-stage process would first retrieve a larger set of candidate documents (e.g., top 20\) using the fast vector search, and then use a more computationally intensive but far more accurate cross-encoder model (e.g., BAAI/bge-reranker-base) to re-score and rank these candidates.

Cross-encoders achieve higher accuracy by performing full attention across the query and each document, capturing semantic nuance that bi-encoders miss.26 By passing only the top 3-5 reranked documents to the LLM, the system provides much more relevant context, leading to more accurate and helpful answers. This pattern is well-supported by LangChain's

ContextualCompressionRetriever, making it straightforward to integrate.28

#### **7.3 Transitioning to an Agentic Architecture**

The final step in this phase is to evolve the application from a chatbot into an AI agent by introducing function-calling capabilities. This involves defining a set of "tools"—Python functions that can perform actions like making calculations or searching the web—and giving the LLM the ability to invoke them. The chat loop would be modified to detect when the model outputs a structured request to call a function, execute that function with the provided arguments, and then feed the result back to the model to generate a final, tool-informed response.

This represents a paradigm shift from passive text generation to active problem-solving. OpenVINO supports models that are fine-tuned for function calling, and frameworks like LangChain provide robust abstractions for building agents that use local OpenVINO models as their reasoning engine.29 This is the next logical step in building more powerful and useful AI applications.

---

## **Part III: Long-Term Vision and Strategic Opportunities**

This part looks beyond the immediate roadmap to consider future directions that can provide a significant and sustainable competitive advantage.

### **Section 8: Exploring Model Customization and Fine-Tuning**

While RAG is highly effective for injecting factual, up-to-date knowledge into an LLM's responses, it is less effective at changing the model's inherent style, tone, or understanding of specialized domain vocabulary. For example, a RAG system can provide a legal document as context, but the LLM might still explain its contents in overly casual language.

For true domain adaptation, fine-tuning is necessary. This process adjusts the model's weights on a smaller, curated dataset, adapting its behavior to a specific task or communication style.32 While the OpenVINO™ Training Extensions project exists 33, the more common path for LLMs involves using standard frameworks like Hugging Face Transformers with Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA, followed by conversion and quantization with Optimum Intel.35

The long-term strategic vision should include creating a feedback loop where high-quality user interactions and key domain-specific documents are used to build a proprietary fine-tuning dataset. This would enable the creation of a custom-tuned version of Phi-3 that not only has access to an organization's knowledge via RAG but also "speaks the language" of the organization, a powerful and defensible differentiator.

### **Section 9: Alternative Deployment Models with OpenVINO™ Model Server (OVMS)**

The current architecture embeds the OpenVINO LLMPipeline directly within the Gradio application. This monolithic deployment model is simple but suffers from tight coupling: the UI and the model cannot be scaled independently, and the application can only serve a single model at a time.

The long-term architectural goal should be to migrate the inference logic to the OpenVINO™ Model Server (OVMS). OVMS is a high-performance, C++ based serving solution designed for production environments. It exposes models via standard gRPC and REST APIs and supports advanced LLM-specific features like continuous batching and prefix caching for improved throughput and latency.18

This migration would transform the system into a modern microservices architecture. The Gradio application would become a lightweight client that makes API calls to a dedicated, scalable OVMS endpoint. This decoupling allows the UI and the AI model to be developed, deployed, and scaled independently. A cluster of OVMS instances, managed by an orchestrator like Kubernetes, could handle heavy inference loads for multiple applications, while only a few instances of the Gradio UI are needed. This architecture is far more robust, scalable, and aligned with modern enterprise IT practices than the current embedded model.18

### **Conclusion and Prioritized Recommendations**

This report has detailed the current state of the Enhanced Phi-3 Chat System, identifying its strengths in NPU optimization and UI design, as well as critical weaknesses related to technical debt and production readiness. To guide its evolution, a clear, prioritized roadmap is essential.

The following actions are recommended:

1. **Immediate Priority (Phase 1):** The highest priority is to stabilize the project's foundation. This requires **unifying the architecture** by removing the legacy monolithic script, **refactoring all legacy naming conventions** to be model-agnostic, and **implementing a CI pipeline with automated testing**. These actions are non-negotiable for ensuring the health, maintainability, and future of the project.  
2. **Mid-Term Priority (Phase 2):** Once the foundation is stable, the focus should shift to production readiness. This involves **containerizing the application with Docker**, establishing a **Continuous Deployment (CD) pipeline** to automate releases, and **enhancing the UI with interactive controls** for generation parameters. These steps will make the application robust, deployable, and more useful.  
3. **Long-Term Priority (Phase 3):** With a production-ready application, the focus can turn to state-of-the-art AI capabilities. This includes **evolving the RAG system with advanced document parsing and cross-encoder reranking** to dramatically improve its accuracy, and beginning the **transition to an agentic architecture** with function-calling capabilities.  
4. **Strategic Vision:** Concurrently with the phased roadmap, the project should plan for two major strategic initiatives: investigating **model fine-tuning** for deep domain adaptation and architecting a future migration to the **OpenVINO™ Model Server (OVMS)** to achieve enterprise-grade scalability and a true microservices architecture.

By systematically executing this roadmap, the project can successfully transition from an impressive but fragile prototype into a robust, scalable, and powerful AI application poised for long-term success.

#### **Referenzen**

1. combined\_openvinodev\_folder.txt  
2. Announcing OpenVINO™ 2025.2: New Models, Generative AI Pipelines, and Performance Improvements \- Medium, Zugriff am August 11, 2025, [https://medium.com/openvino-toolkit/announcing-openvino-2025-2-new-models-generative-ai-pipelines-and-performance-improvements-9e4e46335db3](https://medium.com/openvino-toolkit/announcing-openvino-2025-2-new-models-generative-ai-pipelines-and-performance-improvements-9e4e46335db3)  
3. Releases · openvinotoolkit/openvino \- GitHub, Zugriff am August 11, 2025, [https://github.com/openvinotoolkit/openvino/releases](https://github.com/openvinotoolkit/openvino/releases)  
4. Release Notes for Intel Distribution of OpenVINO Toolkit 2025.2, Zugriff am August 11, 2025, [https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-2.html](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-2.html)  
5. Quickstart \- Gradio, Zugriff am August 11, 2025, [https://www.gradio.app/guides/quickstart](https://www.gradio.app/guides/quickstart)  
6. Introduction to Gradio Blocks \- Hugging Face LLM Course, Zugriff am August 11, 2025, [https://huggingface.co/learn/llm-course/chapter9/7](https://huggingface.co/learn/llm-course/chapter9/7)  
7. OpenVINO GenAI, Zugriff am August 11, 2025, [https://openvinotoolkit.github.io/openvino.genai/](https://openvinotoolkit.github.io/openvino.genai/)  
8. OpenVINO Release Notes — OpenVINO™ documentation — Version(2024), Zugriff am August 11, 2025, [https://docs.openvino.ai/2024/about-openvino/release-notes-openvino.html](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino.html)  
9. Release Notes for Intel Distribution of OpenVINO Toolkit 2024.5, Zugriff am August 11, 2025, [https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2024-5.html](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2024-5.html)  
10. Pytest Coverage Comment · Actions · GitHub Marketplace, Zugriff am August 11, 2025, [https://github.com/marketplace/actions/pytest-coverage-comment](https://github.com/marketplace/actions/pytest-coverage-comment)  
11. Setting Up Continuous Integration with GitHub Actions for a Python Project \- Vuyisile Ndlovu, Zugriff am August 11, 2025, [https://vuyisile.com/setting-up-continuous-integration-with-github-actions-for-a-python-project/](https://vuyisile.com/setting-up-continuous-integration-with-github-actions-for-a-python-project/)  
12. Testing your Python Project with GitHub Actions | by Marc Wouts | TDS Archive | Medium, Zugriff am August 11, 2025, [https://marc-wouts.medium.com/testing-your-python-project-with-github-actions-ec9bf82b20dc](https://marc-wouts.medium.com/testing-your-python-project-with-github-actions-ec9bf82b20dc)  
13. openvino/ubuntu22\_dev \- Docker Image, Zugriff am August 11, 2025, [https://hub.docker.com/r/openvino/ubuntu22\_dev](https://hub.docker.com/r/openvino/ubuntu22_dev)  
14. Install Intel® Distribution of OpenVINO™ Toolkit From a Docker Image, Zugriff am August 11, 2025, [https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-docker-linux.html](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-docker-linux.html)  
15. OpenVINO on Intel Ultra NPU \- Dockerfile \- HackMD, Zugriff am August 11, 2025, [https://hackmd.io/@Otj6UinLQLiYTva4YngKuw/SyVFNztj6](https://hackmd.io/@Otj6UinLQLiYTva4YngKuw/SyVFNztj6)  
16. Deploying Gradio With Docker, Zugriff am August 11, 2025, [https://www.gradio.app/guides/deploying-gradio-with-docker](https://www.gradio.app/guides/deploying-gradio-with-docker)  
17. Deploy a custom Docker image for Data Science project \- Gradio sketch recognition app (Part 1\) \- OVHcloud Blog, Zugriff am August 11, 2025, [https://blog.ovhcloud.com/deploy-a-custom-docker-image-for-data-science-project-gradio-sketch-recognition-app-part-1/](https://blog.ovhcloud.com/deploy-a-custom-docker-image-for-data-science-project-gradio-sketch-recognition-app-part-1/)  
18. openvinotoolkit/model\_server: A scalable inference server for models optimized with OpenVINO \- GitHub, Zugriff am August 11, 2025, [https://github.com/openvinotoolkit/model\_server](https://github.com/openvinotoolkit/model_server)  
19. ChatInterface \- Gradio Docs, Zugriff am August 11, 2025, [https://www.gradio.app/docs/gradio/chatinterface](https://www.gradio.app/docs/gradio/chatinterface)  
20. Blocks And Event Listeners \- Gradio, Zugriff am August 11, 2025, [https://www.gradio.app/guides/blocks-and-event-listeners](https://www.gradio.app/guides/blocks-and-event-listeners)  
21. RAG — Three Python libraries for Pipeline-based PDF parsing \- AI Bites, Zugriff am August 11, 2025, [https://www.ai-bites.net/rag-three-python-libraries-for-pipeline-based-pdf-parsing/](https://www.ai-bites.net/rag-three-python-libraries-for-pipeline-based-pdf-parsing/)  
22. RAG \+ LlamaParse: Advanced PDF Parsing for Retrieval | by Ryan Siegler | KX Systems, Zugriff am August 11, 2025, [https://medium.com/kx-systems/rag-llamaparse-advanced-pdf-parsing-for-retrieval-c393ab29891b](https://medium.com/kx-systems/rag-llamaparse-advanced-pdf-parsing-for-retrieval-c393ab29891b)  
23. Simple Ways to Parse PDFs for Better RAG Systems | by kirouane Ayoub | GoPenAI, Zugriff am August 11, 2025, [https://blog.gopenai.com/simple-ways-to-parse-pdfs-for-better-rag-systems-82ec68c9d8cd](https://blog.gopenai.com/simple-ways-to-parse-pdfs-for-better-rag-systems-82ec68c9d8cd)  
24. How to Efficiently Parse Large PDF and DOCX Files (in GBs) for Embeddings Without Loading Fully in Memory? : r/Rag \- Reddit, Zugriff am August 11, 2025, [https://www.reddit.com/r/Rag/comments/1gjz2dj/how\_to\_efficiently\_parse\_large\_pdf\_and\_docx\_files/](https://www.reddit.com/r/Rag/comments/1gjz2dj/how_to_efficiently_parse_large_pdf_and_docx_files/)  
25. Docling \- GitHub Pages, Zugriff am August 11, 2025, [https://docling-project.github.io/docling/](https://docling-project.github.io/docling/)  
26. The aRt of RAG Part 3: Reranking with Cross Encoders | by Ross Ashman (PhD) | Medium, Zugriff am August 11, 2025, [https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669](https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669)  
27. Rerankers and Two-Stage Retrieval \- Pinecone, Zugriff am August 11, 2025, [https://www.pinecone.io/learn/series/rag/rerankers/](https://www.pinecone.io/learn/series/rag/rerankers/)  
28. Cross Encoder Reranker | 🦜️🔗 LangChain, Zugriff am August 11, 2025, [https://python.langchain.com/docs/integrations/document\_transformers/cross\_encoder\_reranker/](https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/)  
29. Create ReAct Agent using OpenVINO and LangChain, Zugriff am August 11, 2025, [https://docs.openvino.ai/2024/notebooks/llm-agent-react-langchain-with-output.html](https://docs.openvino.ai/2024/notebooks/llm-agent-react-langchain-with-output.html)  
30. OpenVINO | 🦜️ LangChain, Zugriff am August 11, 2025, [https://python.langchain.com/docs/integrations/llms/openvino/](https://python.langchain.com/docs/integrations/llms/openvino/)  
31. Create Function-calling Agent using OpenVINO and Qwen-Agent, Zugriff am August 11, 2025, [https://docs.openvino.ai/2024/notebooks/llm-agent-functioncall-qwen-with-output.html](https://docs.openvino.ai/2024/notebooks/llm-agent-functioncall-qwen-with-output.html)  
32. Optimizing Large Language Models with the OpenVINO™ Toolkit \- Intel® Network Builders, Zugriff am August 11, 2025, [https://builders.intel.com/docs/networkbuilders/optimizing-large-language-models-with-the-openvino-toolkit-1742810892.pdf](https://builders.intel.com/docs/networkbuilders/optimizing-large-language-models-with-the-openvino-toolkit-1742810892.pdf)  
33. OpenVINO Training Extensions download | SourceForge.net, Zugriff am August 11, 2025, [https://sourceforge.net/projects/openvino-train-ext.mirror/](https://sourceforge.net/projects/openvino-train-ext.mirror/)  
34. open-edge-platform/training\_extensions: Train, Evaluate, Optimize, Deploy Computer Vision Models via OpenVINO \- GitHub, Zugriff am August 11, 2025, [https://github.com/open-edge-platform/training\_extensions](https://github.com/open-edge-platform/training_extensions)  
35. LLM Instruction-following pipeline with OpenVINO, Zugriff am August 11, 2025, [https://docs.openvino.ai/2024/notebooks/llm-question-answering-with-output.html](https://docs.openvino.ai/2024/notebooks/llm-question-answering-with-output.html)  
36. Fine-Tuning LLMs: A Guide With Examples \- DataCamp, Zugriff am August 11, 2025, [https://www.datacamp.com/tutorial/fine-tuning-large-language-models](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)  
37. Efficient LLM Serving \- OpenVINO™ documentation, Zugriff am August 11, 2025, [https://docs.openvino.ai/2025/model-server/ovms\_docs\_llm\_reference.html](https://docs.openvino.ai/2025/model-server/ovms_docs_llm_reference.html)  
38. Efficiently Serve Large Language Models (LLMs) with OpenVINO™ Model Server \- Intel, Zugriff am August 11, 2025, [https://www.intel.com/content/dam/develop/public/us/en/documents/llm-with-model-server-white-paper.pdf](https://www.intel.com/content/dam/develop/public/us/en/documents/llm-with-model-server-white-paper.pdf)  
39. Manage deep learning models with OpenVINO Model Server | Red Hat Developer, Zugriff am August 11, 2025, [https://developers.redhat.com/articles/2024/07/03/manage-deep-learning-models-openvino-model-server](https://developers.redhat.com/articles/2024/07/03/manage-deep-learning-models-openvino-model-server)