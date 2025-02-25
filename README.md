# Portfolio of non-proprietary projects

## 1. Code Generation
[Code and text for my doctoral dissertation (Praxis)](https://github.com/agnedil/code-generation) - Boosting the Code Generation Capabilities of Small Language Models (SLMs) Using Agents.

## 2. Chat Fine-Tuning of Mistral 7B Model
Chat fine-tuning of the `Mistral 7B` model using the Supervised Fine-Tuning (SFT) method, incorporating Parameter Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LORA), and 4-bit quantization techniques.
- **Technologies**:
	- **Hugging Face Transformers**: state-of-the-art library for NLP tasks developed by Hugging Face.
	- **Other Hugging Face Libraries**: Specifically, `peft`, `trl`, and `load_dataset` for efficient fine-tuning and data loading.
	- **PyTorch**: open-source ML library for deep learning, e.g. computer vision and NLP applications. Originally developed by Meta AI.
- **Fine-Tuned Models** in HuggingFace Hub: [Mistral-7B-openassistant-guanaco](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco) and [Mistral-7B-openassistant-guanaco-v2](https://huggingface.co/agnedil/Mistral-7B-openassistant-guanaco-v2).
- **Google Colab Notebooks**: Access [notebook 1](https://colab.research.google.com/drive/1q7GpzXDlRrvmpCIFWcZg-WLtKcrzFdGn?usp=sharing) and [notebook 2](https://colab.research.google.com/drive/19lYWzMvZAc2cWPojRiPnYIR5Ok62CgFQ?usp=sharing) used to fine-tune the two models. Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.


## 3. Instruction Fine-Tuning of Llama 2 7B Model
Instruction fine-tuning of the `Llama 2 7B` model using the SFT, PEFT, LORA, and 4-bit quantization techniques.
- **Technologies**: PyTorch and Hugging Face libraries `transformers`, `peft`, `trl`, and `load_dataset`.
- **Fine-Tuned Model** in HuggingFace Hub: [llama-2-7b-alpaca-gpt4](https://huggingface.co/agnedil/llama-2-7b-alpaca-gpt4)
- **Video Demo**: A detailed explanation of concepts and code walkthrough available on [YouTube](https://youtu.be/i9Xtmsbc-74).
- **Google Colab Notebook**: Access the notebook used in the video [here](https://drive.google.com/file/d/1xhO3vxluFqUe5RPPvZhbxVfC1cTVPYgb/view?usp=sharing). Note: click the `Open with Google Colaboratory` button at the top of the page if Google Colab doesn’t open automatically.


## 4. Instruction Fine-Tuning of Llama 2 13B Model with DeepSpeed
Example [script](https://github.com/agnedil/fine-tune-with-deepspeed) for instruction fine-tuning of the `Llama 2 13B model` on a single or multiple GPUs. Adjust code to your specific needs.


## 5. Advanced Gradio RAG Application (Uses Llama) Deployed in HuggingFace Spaces
- **Technologies**: Gradio, LangChain, ensemble retriever FAISS + BM25, re-ranking.
- **Web App**: [RAG Demo with Gradio](https://huggingface.co/spaces/agnedil/rag-demo-with-gradio)
- **GitHub Repository**: [rag-demo-with-gradio](https://github.com/agnedil/rag-demo-with-gradio)


## 6. Advanced Multi-Page Streamlit RAG Application (Uses Llama) Deployed in Streamlit Cloud
- **Technologies**: multi-page Streamlit app, LangChain, ensemble retriever FAISS + BM25, re-ranking, chat history.
- **Web App**: [Streamlit App](https://llm-rag.streamlit.app/)
- **GitHub Repository**: [RAG Demo with Streamlit](https://github.com/agnedil/rag-demo-with-streamlit)
- **Video Demo**: A detailed walkthrough of the app is available on [YouTube](https://youtu.be/CHJo--kQERQ?si=yWyq_0Vr8Igep7mX).


## 7. Other Recent Projects
Check out my [Recent Portfolio](https://github.com/agnedil/Portfolio-Recent) for a glimpse into my other recent non-proprietary projects. Note that being non-proprietary, this code doesn't reflect the complex and multi-faceted nature of projects that I normally complete at work.
