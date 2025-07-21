# STTM: an efficient approach to estimating news impact on stock movement direction

Article link:
https://peerj.com/articles/cs-1156/

This repo contains code for the article. It includes some limited data as well.

- `core/` contains all python scripts for the article.

- `data/sample/market_series/` contains the original limited sample data provided by the authors.

- `data/raw/` contains raw news and market series data. Not included. Place your raw data here.

Other directories will be created in `data/` during operation.

## Running the program:

`py -m core.run_news_preprocessing` processes your news data and stores output in `data/preprocessed_news/`.

`py -m core.run_lda_modeling` builds LDA models and stores output in `data/sttm_weekly_bows/` and `data/models/`.

`py -m core.run_sttm_lda_computation` builds the STTM index and stores output in `data/sttm_index/`.

The original repo published by the authors is [here](https://github.com/hse-scila/-STTM).
