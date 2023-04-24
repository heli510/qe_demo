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

passages = []

# As dataset, we use Simple Etsy Datasets
data_path = '/path/to/your/data/sample/'
file_list = os.listdir(data_path)
print(file_list)

for file_name in file_list:
  with open(data_path+file_name, 'r') as EtsyJson:
    print(data_path+file_name)
    for line in EtsyJson:
      data = json.loads(line.strip())
      #passages.append(data['query'])
      passages.append(data['title'])
    print("Sub Passages:", len(passages))

print("Total Passages:", len(passages))

# We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

embedding_cache_path = '/path/to/your/local/etsy-embeddings-gpu-total.pkl'
print("Store file on disc")
with open(embedding_cache_path, "wb") as fOut:
  pickle.dump({'sentences': passages, 'embeddings': corpus_embeddings}, fOut)
