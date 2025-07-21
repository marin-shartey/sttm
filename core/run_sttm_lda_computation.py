import logging
import os

import pandas as pd
from gensim import models
from gensim.models import LdaModel

from core.config import LDA_MODELS_PATH_PARENT, TRAINING_YEARS, NEWS_PUBLISHER, TICKER, TRAINING_MIN_DATE, \
    TRAINING_MAX_DATE
from core.lib.market_data import load_market_data_excel, compute_weekly_returns
from core.lib.sttm_lda import load_weekly_bow, compute_weekly_topic_vectors, align_weekly_topics_and_returns, \
    compute_topic_return_correlations, store_sttm_index
from core.lib.topic_modeling.lda import get_year_range_str

logging.basicConfig(level=logging.INFO)


def _load_data() -> tuple[LdaModel, dict[tuple[int, int], list[list[tuple[int, int]]]], pd.DataFrame]:
    years_range = get_year_range_str(TRAINING_YEARS)
    model_path = os.path.join(LDA_MODELS_PATH_PARENT, NEWS_PUBLISHER.value, f"lda_{years_range}.model")
    lda_model = models.LdaModel.load(model_path)
    weekly_bows = load_weekly_bow()
    market_data = load_market_data_excel(TICKER, TRAINING_MIN_DATE, TRAINING_MAX_DATE)
    return lda_model, weekly_bows, market_data


# TODO: build STTM indices for all market data, pick top 20% according to granger test

def calculate_sttm_lda():
    """
    Computes a topic stream and correlates it with market data to build an STTM index.
    """
    logging.info("Calculating STTM (LDA) ...")
    logging.info("Loading data ...")
    lda_model, weekly_bows, market_data_daily = _load_data()
    logging.info("Calculating topic stream ...")
    weekly_returns = compute_weekly_returns(market_data_daily)
    weekly_topic_vectors = compute_weekly_topic_vectors(weekly_bows, lda_model)
    topic_matrix, returns, aligned_index = align_weekly_topics_and_returns(weekly_topic_vectors, weekly_returns)
    logging.info("Calculating correlations ...")
    correlations = compute_topic_return_correlations(topic_matrix, returns)
    logging.info(f"Storing STTM index for {TICKER} ...")
    sttm_index = topic_matrix @ correlations
    store_sttm_index(sttm_index, aligned_index, TICKER)

    # test_corpus = [dictionary.doc2bow(doc) for _, doc in documents_test] # dictionary used to be corpora.Dictionary(token_lists)
    # test_topic_dists = [lda_model[doc] for doc in test_corpus]  # list of (topic_id, probability) for each doc
    # print(f"Built test topic distributions: {test_topic_dists is not None}")


calculate_sttm_lda()
