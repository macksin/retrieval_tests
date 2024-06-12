import pandas as pd
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('data/WikiQASent.pos.ans.tsv', sep='\t')

# split the data
columns = ['Question', 'Sentence']
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# fit
corpus = train_set['Question'].tolist() + train_set['Sentence'].tolist()
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# predict
corpus = test_set['Question'].tolist() + test_set['Sentence'].tolist()
tfidf_matrix = vectorizer.transform(corpus)

question_vectors = tfidf_matrix[:len(test_set['Question'])]
answer_vectors = tfidf_matrix[len(test_set['Sentence']):]

cosine_similarities = cosine_similarity(question_vectors, answer_vectors)

import pandas as pd
import numpy as np

K = 5  # você pode ajustar esse valor conforme necessário

# Lista para armazenar as métricas
precision_at_k = []
mrr_scores = []

for index, similarities in enumerate(cosine_similarities):
    # Obtendo os índices das respostas ordenadas por similaridade
    sorted_indices = np.argsort(-similarities)
    top_k_indices = sorted_indices[:K]
    
    # Checando quais são as respostas corretas
    correct_answers = [index]  # isso deve ser definido conforme seu dataset
    
    # Calculando Precision@K
    precision = np.sum(np.isin(top_k_indices, correct_answers)) / K
    precision_at_k.append(precision)

# Calculando a média das métricas
mean_precision_at_k = np.mean(precision_at_k)

print(f"Mean Precision@{K}: {mean_precision_at_k}")

## bm-25
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split(" ") for doc in test_set["Sentence"]]
bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = test_set['Question'].tolist()[0].split(" ")

precision_at_k = []
for idx, query in enumerate(test_set['Question'].tolist()):
    tokenized_query = query.split(" ")
    retrieved_documents = bm25.get_top_n(tokenized_query, test_set["Sentence"].tolist(), n=5)
    precision_at_k.append(1 if test_set["Sentence"].iloc[idx] in retrieved_documents else 0)
print(f"Mean Precision@{K}: {np.mean(precision_at_k)}")