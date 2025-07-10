import gensim.downloader as api

def load_word2vec():
    w2v = api.load("word2vec-google-news-300")
    return {word: w2v[word] for word in w2v.index_to_key}
