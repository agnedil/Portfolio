# Non-Proprietary Projects
This repository contains non-proprietary projects completed outside of my employment.


## 1. Agents
* [Implementing different types of agents using multiple agentic frameworks](https://github.com/agnedil/Portfolio-Recent/tree/main/01-Agents)


## 2. RAG
* [Multiple RAG implementations using LangChain and LangGraph frameworks](https://github.com/agnedil/Portfolio-Recent/tree/main/02-RAG)
* **RAG Using Gradio App in HuggingFace Spaces**
    * **Technologies**: Gradio, LangChain, ensemble retriever FAISS + BM25, re-ranking.
    * **GitHub Repository**: [rag-demo-with-gradio](https://github.com/agnedil/rag-demo-with-gradio)
* **RAG Using Multi-Page Streamlit App**
    * **Technologies**: multi-page Streamlit app, LangChain, ensemble retriever FAISS + BM25, re-ranking, chat history.
    * **GitHub Repository**: [RAG Demo with Streamlit](https://github.com/agnedil/rag-demo-with-streamlit)


## 3. Fine-Tuning LLMs
* [Multi-GPU Fine-Tuning and Inference](https://github.com/agnedil/Portfolio-Recent/tree/main/03-LLM-Fine-Tuning/Multi-GPU)
* Fine-tuning of the `Mistral 7B` model using the Supervised Fine-Tuning (SFT) method, incorporating Parameter Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LORA), and 4-bit quantization techniques. **Technologies**:
	- **Hugging Face Transformers**: state-of-the-art library for NLP tasks developed by Hugging Face.
	- **Other Hugging Face Libraries**: Specifically, `peft`, `trl`, and `load_dataset` for efficient fine-tuning and data loading.
	- **PyTorch**: open-source ML library for deep learning, e.g. computer vision and NLP applications. Originally developed by Meta AI.
    - **Fine-Tuned Models** in HuggingFace Hub: [Mistral-7B-openassistant-guanaco](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco) and [Mistral-7B-openassistant-guanaco-v2](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco-v2).
    - **Google Colab Notebooks**: Access [notebook 1](https://colab.research.google.com/drive/1q7GpzXDlRrvmpCIFWcZg-WLtKcrzFdGn?usp=sharing) and [notebook 2](https://colab.research.google.com/drive/19lYWzMvZAc2cWPojRiPnYIR5Ok62CgFQ?usp=sharing) used to fine-tune the two models. Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.

* Instruction fine-tuning of the `Llama 2 7B` model using the SFT, PEFT, LORA, and 4-bit quantization techniques.
    - **Technologies**: PyTorch and Hugging Face libraries `transformers`, `peft`, `trl`, and `load_dataset`.
    - **Fine-Tuned Model** in HuggingFace Hub: [llama-2-7b-alpaca-gpt4](https://huggingface.co/agnedil/llama-2-7b-alpaca-gpt4)
    - **Video Demo**: A detailed explanation of concepts and code walkthrough available on [YouTube](https://youtu.be/i9Xtmsbc-74).
    - **Google Colab Notebook**: Access the notebook used in the video [here](https://drive.google.com/file/d/1xhO3vxluFqUe5RPPvZhbxVfC1cTVPYgb/view?usp=sharing).
    - Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.

* Instruction Fine-Tuning of Llama 13B Model with DeepSpeed
    - Example [script](https://github.com/agnedil/fine-tune-with-deepspeed) for instruction fine-tuning of the `Llama 2 13B model` on a single or multiple GPUs. Adjust code to your specific needs.


## 4. Recent Projects
[Other non-proprietary projects](https://github.com/agnedil/Portfolio-Recent). Note that being non-proprietary, this code doesn't reflect the complex and multi-faceted nature of projects that I normally complete at work. Content:
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

## 5. Monte Carlo Simulation - Solution for the Monty Hall Problem
[Monte Carlo Simulation for the Monty Hall Problem](https://github.com/agnedil/monte-carlo-for-monty-hall) - empirical experimental confirmation of the famous counter-intuitive answer to the three-door problem: always switch your answer.

## 6. Evaluating Quality of Machine Translation
[Machine Translation Quality Evaluation](https://github.com/agnedil/mt-quality-evaluation) using reference-based and reference-free metrics, as well as LLM-as-a-judge approach.

## 7. Detecting Fake Customer Reviews of Products
* [Methodology and report for detecting fake reviews](https://github.com/agnedil/fake_reviews) about various products on Amazon.com. The methodology is based on machine learning approaches that allow to determine fake reviews automatically. Use case: help potential buyers understand whether the reviews for a specific product are authentic or fake, and what the percentage of fake reviews there is in order to make informed decisions about what products to purchase.


## 8. Code Generation
[Code for my doctoral dissertation](https://github.com/agnedil/code-generation) - Boosting the Code Generation Capabilities of Small Language Models (SLMs) Using Agents. Includes:
* a framework to automatically run and evaluate multiple small language models on 4 code generation evaluation datasets,
* hyperparameter tuning,
* post-processing of generated code,
* SLM fine-tuning,
* experiments with consecutive improvements.


## 9. Python Developer Coding Interview Preparation
[Practical Guide to Prepare for a Python Coding Interview](https://github.com/agnedil/Interview-Prep-Python-Developer) collected from multiple sources. Includes:
* Coding challenges from most of the FAANG companies.
* Algorithms and data structures as examples of Python code.
* Big O notation explained based on practical examples of Python code.
* Other coding resources


## 10. Other Old Project
[Non-proprietary projects completed long time ago](https://github.com/agnedil/Portfolio-Archive). Somewhat outdated and simplistic solutions - keeping just for reference. Content:
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
