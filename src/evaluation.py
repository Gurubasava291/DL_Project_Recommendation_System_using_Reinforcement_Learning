import numpy as np

def precision_at_k(recommended_items, relevant_items, k=5):
    if not relevant_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k).intersection(set(relevant_items)))
    return hits / k

def recall_at_k(recommended_items, relevant_items, k=5):
    if not relevant_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k).intersection(set(relevant_items)))
    return hits / len(relevant_items)

def f1_score_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def ndcg_at_k(recommended_items, relevant_items, k=5):
    if not relevant_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0 and 0-indexed
            
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg
