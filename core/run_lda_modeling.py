from core.config import NEWS_PUBLISHER, TRAINING_YEARS
from core.lib.sttm_lda import store_weekly_bows
from core.lib.topic_modeling.lda import train_lda_model, load_news_tokens, store_model
from core.lib.topic_modeling.topic_modeling import build_bows_with_dates


def build_lda_model_for_sttm():
    training_documents = load_news_tokens(NEWS_PUBLISHER, TRAINING_YEARS)
    dictionary, training_corpus_with_dates = build_bows_with_dates(training_documents)
    store_weekly_bows(training_corpus_with_dates)
    training_corpus = [bow for _, bow in training_corpus_with_dates]
    # store_dictionary_and_corpus(dictionary, corpus_train, NEWS_PUBLISHER, TRAINING_YEARS)

    lda_model = train_lda_model(training_corpus, dictionary, num_topics=20, passes=10)
    store_model(lda_model, NEWS_PUBLISHER, TRAINING_YEARS)
    lda_model.print_topics()


build_lda_model_for_sttm()
