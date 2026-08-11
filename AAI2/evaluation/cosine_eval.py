from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer("all-MiniLM-L6-v2")

reference="""
Please provide order details and we will arrange replacement.
"""

def cosine(response):

    emb=model.encode([reference,response])

    return cosine_similarity([emb[0]],[emb[1]])[0][0]