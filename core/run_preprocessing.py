import os

import nltk

from lib.news_preprocessing import NewsPublisher, process_news_publisher_dir, \
    write_yearly_data

# Paths relative to project root
RAW_DATA_PATH_PARENT = "./data/raw/news"
OUTPUT_PATH_PARENT = "./data/preprocessed/news"
LIMIT = 0  # Number of articles to process (for testing). 0 = no limit.

if __name__ == "__main__":
    """
    Preprocess news articles and write to CSV.
    """
    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # For each news source directory, and each date directory in it, preprocess the articles and write to CSV.
    for publisher_dir_name in os.listdir(RAW_DATA_PATH_PARENT):
        news_publisher = None
        # Check if the news source is familiar so we can apply special parsing
        if isinstance(publisher_dir_name, NewsPublisher):
            news_publisher = NewsPublisher[publisher_dir_name]

        yearly_data = process_news_publisher_dir(RAW_DATA_PATH_PARENT, publisher_dir_name, news_publisher, LIMIT)
        write_yearly_data(OUTPUT_PATH_PARENT, publisher_dir_name, yearly_data)
