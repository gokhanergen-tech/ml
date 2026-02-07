# AI and History

Explore and learn about Machine Learning:  
[General Machine Learning Info](https://www.kaggle.com/code/gkhanergen/general-machine-learning-info)

---

## Creating a Basic Perceptron

Learn about perceptrons, the building blocks of Artificial Neural Networks:  
[Basic Perceptron Examples](https://www.kaggle.com/code/gkhanergen/creating-a-basic-perceptron)

---

## 🖼️ Image-to-Caption Models

Models that generate captions or textual descriptions from images, combining vision and language tasks.

- **[Hugging Face Transformers — Image Captioning](https://huggingface.co/docs/transformers/v4.47.0/tasks/image_captioning)**  
  Learn how models generate captions from images. The Hugging Face documentation offers detailed insights into various **image captioning models** and tasks.  

- **[Microsoft Kosmos-2 Patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224)**  
  A multimodal model capable of both **vision and language tasks**.

- **[CLIP (Contrastive Language-Image Pretraining)](https://github.com/openai/CLIP)**  
  CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image

- **[ViLT (Vision-and-Language Transformer)](https://github.com/dandelin/vilt)**  
 Code for the ICML 2021 (long talk) paper: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"

## BERT and Transformer Models
**BERT (Bidirectional Encoder Representations from Transformers)** is a **pretrained transformer-based model** developed by **Google AI** in 2018.
Pretrained language models for embeddings, NLP tasks, and semantic understanding.

- **[Turkish BERT — Embeddings for Semantic Search](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)**  
  Converts text into embeddings to use for tasks like clustering, semantic search, or similarity comparison.  
  **Use case:** Turkish NLP projects, semantic search, and text clustering.

- **[BERT-base-uncased](https://huggingface.co/bert-base-uncased)**  
  Pretrained model on English language using a masked language modeling (MLM) objective.
  
- **[distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)**   
  **Use case:** You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. 

- **[multilingual-bert (mBERT)](https://huggingface.co/bert-base-multilingual-cased)**  
  Pretrained on 104 languages.  

- **[Sentence-BERT (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**  
  This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

- **[FinBERT](https://huggingface.co/ProsusAI/finbert)**  
  FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and    thereby fine-tuning it for financial sentiment classification. 

- **[BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1)**  

- **[RoBERTa-base](https://huggingface.co/roberta-base)**  
  Pretrained model on English language using a masked language modeling (MLM) objective.  

## Image Generation Tools
Tools and models that generate images from text prompts using open‑source AI.

- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)**  
  Popular open‑source text‑to‑image model producing high‑quality images based on descriptions.  
  **Use case:** Artistic content, concept art, prototyping images for ML datasets. 

- **[AUTOMATIC1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**  
  A powerful web user interface for running Stable Diffusion locally with advanced features like in‑painting, out‑painting, and custom samplers.  
  **Use case:** Local interactive generation + extensive customization.

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**  
  Node‑based image generation workflow builder using Stable Diffusion models.  
  **Use case:** Custom pipelines with nodes for complex generative workflows. 

- **[Fooocus](https://github.com/lllyasviel/Fooocus)**  
  Streamlined open‑source generator built on Stable Diffusion XL, easy to use out‑of‑the‑box.  
  **Use case:** Simple, Midjourney‑like generation without complex setup. 

- ** https://www.openjourney.art/

- **[Waifu Diffusion](https://huggingface.co/hakurei/waifu-diffusion)**  
  Anime‑style version of Stable Diffusion fine‑tuned on high‑quality anime images.  
  **Use case:** Anime and stylized artwork generation.

- **[Waifu Diffusion v1.4](https://huggingface.co/hakurei/waifu-diffusion-v1-4)**  
  Updated variant of the Waifu Diffusion model with refined anime results.  
  **Use case:** Anime‑style character and scene generation.

- **[DreamShaper](https://huggingface.co/Lykon/DreamShaper)**  
  Artistic text‑to‑image model built on Stable Diffusion designed for creative outputs.  
  **Use case:** Versatile art generation from text.

## Optimization Algorithm Tools

Libraries and frameworks for solving optimization problems, including metaheuristic, evolutionary, and swarm algorithms.

- **[Mealpy](https://github.com/thieu1995/mealpy/tree/master/mealpy)**  
  Python library for metaheuristic optimization algorithms.  
  **Use case:** Solve optimization problems using PSO, GA, DE, WOA, etc.

- **[PyGMO / Pagmo](https://esa.github.io/pygmo2/)**  
  Parallel Global Multiobjective Optimizer; provides evolutionary algorithms, swarm intelligence, and parallel computation.  
  **Use case:** Large-scale multi-objective optimization problems.

- **[DEAP](https://github.com/DEAP/deap)**  
  Distributed Evolutionary Algorithms in Python.  
  **Use case:** Evolutionary computation, genetic algorithms, and multi-objective optimization.  

- **[Inspyred](https://github.com/aarongarrett/inspyred)**  
  About Python library for bio-inspired computational intelligence 

- **[Optuna](https://optuna.org/)**  
  Automatic hyperparameter optimization framework.  
  **Use case:** Optimize machine learning model parameters efficiently.

- **[Nevergrad](https://github.com/facebookresearch/nevergrad)**  
  Gradient-free optimization platform by Facebook Research.  

- **[PySwarms](https://github.com/ljvmiranda921/pyswarms)**  
  Particle Swarm Optimization in Python.  

- **[SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)**  
  Built-in Python library for mathematical and numerical optimization.  
  **Use case:** Standard optimization algorithms (linear, nonlinear, constrained).

## Annotation Tools

Tools for labeling images, videos, and text for computer vision and NLP tasks.

- **[CVAT](https://github.com/cvat-ai/cvat)**  
  Interactive video and image annotation tool for computer vision tasks.

- **[LabelImg](https://github.com/heartexlabs/labelImg)**  
  Image annotation tool for bounding boxes (object detection datasets).

- **[Label Studio](https://github.com/heartexlabs/label-studio)**  
  Versatile annotation tool supporting images, text, audio, video, and time-series data.

- **[VoTT (Visual Object Tagging Tool)](https://github.com/microsoft/VoTT)**  
  Microsoft’s tool for image and video labeling, supports exporting in multiple ML formats.

- **[RectLabel](https://rectlabel.com/)** (macOS)  
  Image annotation for object detection, segmentation, and bounding boxes.

- **[Doccano](https://github.com/doccano/doccano)**  
  Open-source text annotation tool for NLP tasks (text classification, sequence labeling, etc.).

- **[VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)**  
  Simple and lightweight browser-based image and video annotation tool.


## Useful Python Tools

### ML & Data Science Libraries
- `optuna` — Hyperparameter optimization framework  
- `pytorch` — Deep learning library  
- `tensorflow` — Deep learning library  
- `keras` — High-level neural networks API  
- `sklearn` — Machine learning models  
- `catboost`, `xgboost`, `lightGBM` — Boosting algorithms  

### Data Processing
- `numpy` — Numerical computations  
- `pandas` — Data manipulation  
- `polars` — Fast DataFrame library  
- `jax` — GPU-accelerated numerical computations with automatic differentiation  

### GUI Development
- `DearPyGUI`, `PyQt`, `Kivy` — GUI application development  

### Web & API
- `FastAPI` — High-performance web APIs  

### Dashboards
- [Python Dashboard Frameworks](https://www.planeks.net/python-dashboard-development-framework/)  
  Guide to building dashboards in Python.

# Awesome LLM Repositories

This repository is curated to collect **Large Language Models (LLMs)** and related tools, frameworks, and educational resources.  
You can use it as a reference for your projects or for learning purposes.

---

##  General and Useful “Awesome” LLM Repos

- **[Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)**  
  A curated list of LLM resources.

- **[LLMs](https://github.com/Ggasyd/LLMs)**  
  A comprehensive list of LLMs and their models.

---

##  Popular and Notable LLM / Tool Repos

### Models, Tools, and Frameworks
- **[Transformers (Hugging Face)](https://github.com/huggingface/transformers)**  
  A core library for working with LLMs.

- **[Ollama](https://github.com/ollama/ollama)**  
  LLM infrastructure with support for multiple models.

- **[LangChain](https://github.com/langchain-ai/langchain)**  
  Chaining tasks and building agents using LLMs.

- **[open-webui](https://github.com/open-webui/open-webui)**  
  User interface for interacting with LLMs.

---

## Learning and Example Projects

- **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)**  
  Step-by-step guide to building LLMs (PyTorch).

- **[gpt4free](https://github.com/xtekky/gpt4free)**  
  Open-source models and chat infrastructure.

- **[LLM4Rec](https://github.com/Ruyu-Li/LLM4Rec)**  
  LLMs for recommendation systems.

---

## Other Useful Repos

- **[Awesome-local-LLM](https://github.com/rafska/Awesome-local-LLM)**  
  Guide to running LLMs locally.

- **[RAGLight](https://github.com/Bessouat40/RAGLight)**  
  CLI tool to chat with GitHub repositories.

---

## GitHub Topic Pages

- [large-language-models](https://github.com/topics/large-language-models)  
- [llm](https://github.com/topics/llm)

---

# Resources:
1. https://www.researchgate.net/publication/363196371_Recent_Trends_in_Artificial_Intelligence_for_Subsurface_Geothermal_Applications
2. https://www.historyofinformation.com/detail.php?entryid=782
3. https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/
4. https://www.geeksforgeeks.org/ml-linear-regression/
5. https://home.csulb.edu/~cwallis/artificialn/History.htm
