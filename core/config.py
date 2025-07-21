from datetime import date
from enum import Enum


class NewsPublisher(str, Enum):
    KOMMERSANT = "kommersant"
    RIA_NOVOSTI = "ria_novosti"


RAW_NEWS_PATH_PARENT = "./data/raw/news"
MARKET_SERIES_PATH_PARENT = "./data/raw/market_series"
PREPROCESSED_NEWS_PATH_PARENT = "./data/preprocessed_news"
MYSTEM_PATH = "./core/lib/mystem"
LDA_MODELS_PATH_PARENT = "./data/models/lda"
TRAINING_YEARS = [2021, 2022, 2023]
TEST_YEAR = TRAINING_YEARS[-1] + 1 if True else None
NEWS_PUBLISHER = NewsPublisher.KOMMERSANT
WEEKLY_BOWS_PATH_PARENT = "./data/sttm_weekly_bows"
STTM_INDICES_PATH_PARENT = "./data/sttm_indices"
TICKER="MOEX" # TODO: expand to other indices

# Adjust to match the exact date range which the news data covers.
TRAINING_MIN_DATE = date(2021, 1, 4)
TRAINING_MAX_DATE = date(2023, 12, 29)
