import gensim.downloader as api

from gensim.models import Word2Vec


class Word2VecModelTrainer:
    def train_on_wiki(self):
        data = api.load('wiki-english-20171001')
        print(len(data))
        word2vec_model = Word2Vec(data)
        word2vec_model.save('word2vec/models/model.bin')

    def train(self, csv_file_name):
        pass
        #data_reader = DataReader(csv_file_name)
        #ids = data_reader.get_ids()
        #train_ids, test_ids = train_test_split(ids, random_state=0)
        #word2vec_model.build_vocab(data_reader.get_texts(train_ids), update=True)
        #word2vec_model.train(data_reader.get_texts(train_ids))
