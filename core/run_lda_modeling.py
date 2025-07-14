from core.lib.data_loader import load_news_tokens_for_year
from core.lib.news_preprocessing import NewsPublisher
from core.lib.topic_modeling.lda import train_lda_model
from core.lib.topic_modeling.topic_modeling import build_bow_corpus

documents_train = load_news_tokens_for_year(2021, NewsPublisher.KOMMERSANT)
documents_test = load_news_tokens_for_year(2022, NewsPublisher.KOMMERSANT)


def build_lda_model():
    dictionary, corpus_train = build_bow_corpus(documents_train)
    lda_model = train_lda_model(corpus_train, dictionary, num_topics=20, passes=10)

    corpus_test = [dictionary.doc2bow(doc) for doc in documents_test]
    test_topic_dists = [lda_model[doc] for doc in corpus_test]  # list of (topic_id, probability) for each doc
    lda_model.print_topics()


build_lda_model()
