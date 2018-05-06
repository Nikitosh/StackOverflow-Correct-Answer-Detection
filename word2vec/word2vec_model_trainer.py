import os.path

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from models.data_reader import DataReader
from models.utils import get_word2vec_model_path
from word2vec.wiki_sentences import WikiSentences

class Word2VecModelTrainer:
    WIKI_MODEL_PATH = 'word2vec/models/wiki_model.bin'

    def train_on_wiki(self):
        sentences = WikiSentences('wiki-english-20171001')
        word2vec_model = Word2Vec(sentences, size=300)
        word2vec_model.save(Word2VecModelTrainer.WIKI_MODEL_PATH)

    def train(self, csv_file_name):
        model_path = get_word2vec_model_path(csv_file_name)
        if os.path.isfile(model_path):
            return
        if not os.path.isfile(Word2VecModelTrainer.WIKI_MODEL_PATH):
            self.train_on_wiki()
        word2vec_model = Word2Vec.load(Word2VecModelTrainer.WIKI_MODEL_PATH)
        data_reader = DataReader(csv_file_name)
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        word2vec_model.min_count = 3
        word2vec_model.build_vocab(data_reader.get_texts_as_lists(train_ids), update=True)
        word2vec_model.train(data_reader.get_texts_as_lists(train_ids), epochs=word2vec_model.iter,
                             total_examples=word2vec_model.corpus_count)
        word2vec_model.save(model_path)
