# Portfolio of Non-Proprietary Projects
This repository includes only non-proprietary projects completed outside of my employment. Please keep in mind that the time available for such work is limited.

## 1. Code Generation
[Code for and text of my doctoral dissertation](https://github.com/agnedil/code-generation) - Boosting the Code Generation Capabilities of Small Language Models (SLMs) Using Agents. Includes:
* a framework to automatically run and evaluate multiple small language models on 4 code generation datasets,
* hyperparameter tuning,
* post-processing of generated code,
* SLM fine-tuning,
* experiments with consecutive improvements.

## 2. Agents
* [Various types of agents implemented](https://github.com/agnedil/Portfolio-Recent/tree/main/01-Agents)

## 3. RAG
* [RAG implemented using Langchain and Langgraph frameworks](https://github.com/agnedil/Portfolio-Recent/tree/main/02-RAG)
* Advanced RAG - Gradio App in HuggingFace Spaces
** **Technologies**: Gradio, LangChain, ensemble retriever FAISS + BM25, re-ranking.
** **Web App**: [RAG Demo with Gradio](https://huggingface.co/spaces/agnedil/rag-demo-with-gradio)
** **GitHub Repository**: [rag-demo-with-gradio](https://github.com/agnedil/rag-demo-with-gradio)
* Advanced RAG - Multi-Page Streamlit App
** **Technologies**: multi-page Streamlit app, LangChain, ensemble retriever FAISS + BM25, re-ranking, chat history.
** **Web App**: [Streamlit App](https://llm-rag.streamlit.app/)
** **GitHub Repository**: [RAG Demo with Streamlit](https://github.com/agnedil/rag-demo-with-streamlit)
** **Video Demo**: A detailed walkthrough of the app is available on [YouTube](https://youtu.be/CHJo--kQERQ?si=yWyq_0Vr8Igep7mX).

## 4. Fine-Tuning LLMs
* [Multi-GPU Fine-Tuning and Inference](https://github.com/agnedil/Portfolio-Recent/tree/main/03-LLM-Fine-Tuning/Multi-GPU)
* Fine-tuning of the `Mistral 7B` model using the Supervised Fine-Tuning (SFT) method, incorporating Parameter Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LORA), and 4-bit quantization techniques.
** **Technologies**:
	- **Hugging Face Transformers**: state-of-the-art library for NLP tasks developed by Hugging Face.
	- **Other Hugging Face Libraries**: Specifically, `peft`, `trl`, and `load_dataset` for efficient fine-tuning and data loading.
	- **PyTorch**: open-source ML library for deep learning, e.g. computer vision and NLP applications. Originally developed by Meta AI.
** **Fine-Tuned Models** in HuggingFace Hub: [Mistral-7B-openassistant-guanaco](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco) and [Mistral-7B-openassistant-guanaco-v2](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco-v2).
** **Google Colab Notebooks**: Access [notebook 1](https://colab.research.google.com/drive/1q7GpzXDlRrvmpCIFWcZg-WLtKcrzFdGn?usp=sharing) and [notebook 2](https://colab.research.google.com/drive/19lYWzMvZAc2cWPojRiPnYIR5Ok62CgFQ?usp=sharing) used to fine-tune the two models. Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.
* Instruction fine-tuning of the `Llama 2 7B` model using the SFT, PEFT, LORA, and 4-bit quantization techniques.
** **Technologies**: PyTorch and Hugging Face libraries `transformers`, `peft`, `trl`, and `load_dataset`.
** **Fine-Tuned Model** in HuggingFace Hub: [llama-2-7b-alpaca-gpt4](https://huggingface.co/agnedil/llama-2-7b-alpaca-gpt4)
** **Video Demo**: A detailed explanation of concepts and code walkthrough available on [YouTube](https://youtu.be/i9Xtmsbc-74).
** **Google Colab Notebook**: Access the notebook used in the video [here](https://drive.google.com/file/d/1xhO3vxluFqUe5RPPvZhbxVfC1cTVPYgb/view?usp=sharing). ** Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.
* Instruction Fine-Tuning of Llama 13B Model with DeepSpeed
** Example [script](https://github.com/agnedil/fine-tune-with-deepspeed) for instruction fine-tuning of the `Llama 2 13B model` on a single or multiple GPUs. Adjust code to your specific needs.

## 5. Detecting Fake Customer Reviews of Products
[Methodology and report for detecting fake reviews](https://github.com/agnedil/fake_reviews) about various products on Amazon.com. The methodology is based on machine learning approaches that allow to determine fake reviews automatically. Use case: help potential buyers understand whether the reviews for a specific product are authentic or fake, and what the percentage of fake reviews there is in order to make informed decisions about what products to purchase.

## 6. Python Developer Coding Interview Preparation
[Practical Guide to Prepare for a Python Coding Interview](https://github.com/agnedil/Interview-Prep-Python-Developer) which I collected from multiple sources. Includes:
* Coding challenges from most of the FAANG companies.
* Algorithms and data structures as examples of Python code.
* Big O notation explained based on practical examples of Python code.
* Other coding resources


## 7. Other Recent Projects
Check out my [Recent Portfolio](https://github.com/agnedil/Portfolio-Recent) for a glimpse into my other recent non-proprietary projects. Note that being non-proprietary, this code doesn't reflect the complex and multi-faceted nature of projects that I normally complete at work. Content:
* NLP
    * Participation in two ACL (Association for Computational Linguistics) WASSA 2023 Shared Tasks
    * Prompt Engineering
    * ChatGPT, Gemini and other APIs
    * Transformers
    * Semantic search
    * Deep Learning (LSTMs and other DL models)
    * BoW models
* Other Machine Learning Topics
    * General machine learning algorithms and helper functions
* Certificates
    * Certificates for NLP and DL nano-degrees and other courses 
* Visualization
    * I have used multiple visualization tools: Matplotlib, Plotly, Ggplot2, R Shiny, Tableau, Spotfire, etc.
    * Small example of non-proprietary visualization project using D3.js: **website** at https://agnedil.github.io/; **code** at https://github.com/agnedil/agnedil.github.io


## 8. Other Old Project
[Non-proprietary projects completed long time ago.](https://github.com/agnedil/Portfolio-Archive). Some of this code may seem simplistic, but I would still like to keep it for reference because there are code snippets containing interesting solutions. Content:
* NLP
    * Embeddings
    * Topic modeling
    * Conversational AI
    * Other miscellaneous projects
* Machine Learning
    * General machine learning algorithms and helper functions
* Computer Vision
    * Semantic segmentation, image denoising, classification and other image analysis methods
* Cloud Technologies
    * Hadoop, MapReduce, MLlib, Giraph, Apache Spark, etc.
