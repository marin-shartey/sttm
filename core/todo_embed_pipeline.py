from gensim.models import Word2Vec
import numpy as np

def train_word2vec(token_lists, size=300):
    model = Word2Vec(sentences=token_lists, size=size, window=5, min_count=1, workers=1, seed=7)
    return {word: model.wv[word] for word in model.wv.index_to_key}

def embed_word2vec(x, w2v_dict):
    ar = []
    for i in x:
        vecs = [w2v_dict[j] for j in i if j in w2v_dict]
        if vecs:
            ar.append(np.mean(vecs, axis=0))
    return np.mean(ar, axis=0) if ar else np.zeros(next(iter(w2v_dict.values())).shape)
