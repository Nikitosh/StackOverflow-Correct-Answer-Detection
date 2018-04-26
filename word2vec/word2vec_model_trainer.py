from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split

from models.data_reader import DataReader


class Word2VecModelTrainer:
    def train(self, csv_file_name):
        data_reader = DataReader(csv_file_name)
        word2vec_model = Word2Vec()
        word2vec_vectors = KeyedVectors.load_word2vec_format('word2vec/models/google_news_vectors.bin', binary=True)
        word2vec_model.wv = word2vec_vectors
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        word2vec_model.build_vocab(data_reader.get_texts(train_ids), update=True)
        word2vec_model.train(data_reader.get_texts(train_ids))
        word2vec_model.save('word2vec/models/google_news_updated.bin')