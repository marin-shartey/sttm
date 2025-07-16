import os

import nltk

from core.config import RAW_NEWS_PATH_PARENT
from core.lib.news_preprocessing import NewsPublisher, process_news_publisher_dir, \
    write_yearly_data

LIMIT = 0  # Number of articles to process (used for testing). 0 = no limit.


def preprocess_news():
    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # For each news source directory, and each date directory in it, preprocess the articles and write to CSV.
    for publisher_dir_name in os.listdir(RAW_NEWS_PATH_PARENT):
        news_publisher = None
        # Check if the news source is familiar so we can apply special parsing
        if publisher_dir_name in NewsPublisher._value2member_map_:
            news_publisher = NewsPublisher._value2member_map_[publisher_dir_name]

        yearly_data = process_news_publisher_dir(publisher_dir_name, news_publisher, LIMIT)
        write_yearly_data(publisher_dir_name, yearly_data)


preprocess_news()
