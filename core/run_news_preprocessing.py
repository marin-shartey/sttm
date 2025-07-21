import logging
import os

import nltk

from core.config import RAW_NEWS_PATH_PARENT
from core.lib.news_data_manager import write_yearly_data
from core.lib.news_preprocessing import NewsPublisher, process_news_publisher_dir

logging.basicConfig(level=logging.DEBUG)

LIMIT = 0  # Number of articles to process (used for testing). 0 = no limit.


def preprocess_news():
    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # For each news source directory, and each date directory in it, preprocess the articles and write to CSV.
    dir_list = os.listdir(RAW_NEWS_PATH_PARENT)
    count = 1
    for publisher_dir_name in dir_list:
        logging.info(f"Processing {publisher_dir_name} ({count}/{len(dir_list)})")
        count += 1
        news_publisher = None
        # Check if the news source is familiar so we can apply special parsing
        if publisher_dir_name in NewsPublisher._value2member_map_:
            news_publisher = NewsPublisher._value2member_map_[publisher_dir_name]

        yearly_data = process_news_publisher_dir(publisher_dir_name, news_publisher, LIMIT)
        write_yearly_data(publisher_dir_name, yearly_data)


preprocess_news()
