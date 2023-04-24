# We also compare the results to lexical search (keyword search). Here, we use 
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np


# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc
 
# This function will search all wikipedia articles for passages that
# answer the query
def search(query):
    print("Input query:", query)
    total_qe = []

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    #print("Top-10 lexical search (BM25) hits")
    qe_string = []
    for hit in bm25_hits[0:1000]:
      if passages[hit['corpus_id']].replace("\n", " ") not in qe_string:
        qe_string.append(passages[hit['corpus_id']].replace("\n", ""))

    sub_string = []
    for item in qe_string:
      for sub_item in item.split(","):
        sub_string.append(sub_item)
    total_qe.append(sub_string)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cuda()
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-10 hits from bi-encoder
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    qe_string = []
    for hit in hits[0:1000]:
      if passages[hit['corpus_id']].replace("\n", " ") not in qe_string:
        qe_string.append(passages[hit['corpus_id']].replace("\n", ""))
    total_qe.append(qe_string)

    # Output of top-10 hits from re-ranker
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    qe_string = []
    for hit in hits[0:1000]:
      if passages[hit['corpus_id']].replace("\n", " ") not in qe_string:
        qe_string.append(passages[hit['corpus_id']].replace("\n", ""))
    total_qe.append(qe_string)

    # Total Results
    total_qe.append(qe_string)
    print("E-Commerce Query Expansion Results: \n")
    print(total_qe)

# -- main function -- #
tokenized_corpus = []
for passage in tqdm(passages):
    tokenized_corpus.append(bm25_tokenizer(passage))
bm25 = BM25Okapi(tokenized_corpus)
search(query = "gift")



