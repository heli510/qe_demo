import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import pickle

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")

#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256    #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Load pre-computed embeddings from disc")
embedding_cache_path = '/path/to/your/local/etsy-embeddings-gpu.pkl'

with open(embedding_cache_path, "rb") as fIn:
  cache_data = pickle.load(fIn)
  passages = cache_data['sentences']
  corpus_embeddings = cache_data['embeddings']
