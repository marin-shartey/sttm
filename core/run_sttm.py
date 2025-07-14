from navec import Navec

from core.lib.sample_data_loader import load_news, load_returns
from core.todo_embed_pipeline import embed_word2vec
from core.lib.embeddings_ml_models import train_test_split, get_expanding_predictions

from sklearn.ensemble import GradientBoostingClassifier

from core.lib.load_embeddings import load_word2vec

# Load data
news = load_news()
returns = load_returns()

w2v_dict = load_word2vec()
navec_dict = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')

# Weekly aggregated embeddings
emb = news.set_index('issuedate').resample('1W')['preproc'].apply(lambda x: embed_word2vec(x, w2v_dict))

# Split and train
train_dict, test_dict = train_test_split(emb, returns)
preds = get_expanding_predictions(GradientBoostingClassifier(), train_dict, test_dict)

# Save predictions
preds.to_csv("predictions.csv")
