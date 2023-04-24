# qe_demo
Query Expansion Demo for e-commerce

## Installation

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with requestment.txt**

streamlit==0.82.0

streamlit_tags

pyarrow

keytotext

opencv-python-headless

sentence-transformers

rank_bm25

yake


**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Getting Started

1. Run encoder.py - For encoding all passages from datasets (only support json format input files). And the embedding files will be stored by the name of "*.pkl"

2. Run loader.py - For loading pre-computed embeddings

3. Run model_run.py - For start our basic pre-train model and outcome the expansion results.
