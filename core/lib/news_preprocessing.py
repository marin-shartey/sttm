import logging
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm import tqdm

from core.config import MYSTEM_PATH, RAW_NEWS_PATH_PARENT, PREPROCESSED_NEWS_PATH_PARENT

logging.basicConfig(level=logging.DEBUG)

class NewsPublisher(str, Enum):
    KOMMERSANT = "kommersant"
    RIA_NOVOSTI = "ria_novosti"


russian_stopwords = set(stopwords.words("russian"))
punctuation = set(string.punctuation)


def extract_date_from_path(file_path: str) -> str or None:
    date_str = Path(file_path).parent.name
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        return None


def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines if line.strip()]


def merge_lines_with_periods(lines) -> str:
    for i, line in enumerate(lines):
        if not line.endswith('.'):
            lines[i] = line + '.'
    return " ".join(lines)


def parse_article(lines) -> str:
    """
    Parse generic article. Join all lines into one string.
    """
    clean_lines = [line.strip() for line in lines if line.strip()]
    return merge_lines_with_periods(clean_lines)


def parse_kommersant_article(lines) -> str:
    """
    Parse kommersant.ru article. Skip the first line and join the rest into one string.
    Expected format:
    - line 0 = datetime
    - line 1 = title
    - line 2 optional = subtitle
    - rest = article
    """
    if len(lines) < 2:
        # Empty article only has time on one line
        return ""

    # Join all lines except first, add periods if absent.
    clean_lines = [line.strip() for line in lines[1:] if line.strip()]
    return merge_lines_with_periods(clean_lines)


def lemmatize_text_mystem_b(text: str) -> list:
    os.environ.update({'MYSTEM_BIN': MYSTEM_PATH})
    mystem = Mystem()
    lemmas = mystem.lemmatize(text=text)
    return lemmas


def normalize_and_filter_tokens(tokens: list[str]) -> list[str]:
    """
    Normalizes tokens by stripping whitespace, filtering out non-Cyrillic words, punctuation, and stopwords.
    """
    pattern = re.compile(r'^[а-яё]+$')

    return [
        t for t in (token.strip() for token in tokens)
        if pattern.match(t) and t not in russian_stopwords and t not in punctuation
    ]


def preprocess_single_file(file_path: str, news_publisher: Optional[NewsPublisher]) -> (str, str, str) or None:
    """
    Generate an entry for a raw news file.
    """
    date_str = extract_date_from_path(file_path)
    if date_str is None:
        return None

    lines = read_file_lines(file_path)
    if news_publisher is NewsPublisher.KOMMERSANT:
        article_text = parse_kommersant_article(lines)
    else:
        article_text = parse_article(lines)

    if article_text is None:
        return None

    lemmas = lemmatize_text_mystem_b(article_text)
    normalized_tokens = normalize_and_filter_tokens(lemmas)
    normalized_text = " ".join(normalized_tokens)

    return date_str, article_text, normalized_text


def process_news_publisher_dir(
        publisher_dir_name: str,
        news_publisher: Optional[NewsPublisher],
        limit: int = 0
    ) -> dict[str, list[str]] or None:
    yearly_data = {}
    raw_data_dir = os.path.join(RAW_NEWS_PATH_PARENT, publisher_dir_name)

    counter = 0

    for news_by_date_dir in tqdm(os.listdir(raw_data_dir)):
        date_path = os.path.join(raw_data_dir, news_by_date_dir)
        if not os.path.isdir(date_path):
            continue

        files = [f for f in os.listdir(date_path) if f.endswith('.txt')]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(preprocess_single_file, os.path.join(date_path, f), news_publisher): f for f in
                       files}
            for future in as_completed(futures):
                article_data = future.result()
                if article_data is None:
                    continue
                year = article_data[0][:4]
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(article_data)
                counter += 1
                if 0 < limit <= counter:
                    break

        if 0 < limit <= counter:
            break

    return yearly_data


def write_yearly_data(publisher_dir_name: str, yearly_data: dict[str, list[str]] or None):
    output_dir = os.path.join(PREPROCESSED_NEWS_PATH_PARENT, publisher_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    if yearly_data is None:
        logging.error(">>> No yearly data found for publisher %s", publisher_dir_name)
        return

    for year, data in yearly_data.items():
        df = pd.DataFrame(data, columns=['date', 'text', 'preproc'])
        output_path = os.path.join(output_dir, f"{year}.csv")
        df.to_csv(output_path, index=False)
