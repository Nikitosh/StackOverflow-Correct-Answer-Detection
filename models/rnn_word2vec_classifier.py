import numpy as np
from keras import Sequential
from keras.layers import Dropout, Dense

from gensim.models.word2vec import Word2Vec
from models.utils import transform_sentence_batch_to_vector, lower_text, get_word2vec_model_path, add_lstm_to_model
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class RnnWord2VecClassifier:
    def __init__(self, bidirectional=False, dropout=0.1):
        self.num_features = 300
        self.document_max_num_words = 300
        self.embed_size = 100

        self.word_vectors = None

        self.model = Sequential()
        input_shape = (self.document_max_num_words, self.num_features)
        add_lstm_to_model(self.model, self.embed_size, bidirectional, input_shape, dropout)
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def pretrain(self, data_reader, ids):
        trainer = Word2VecModelTrainer()
        trainer.train(data_reader, ids)
        self.word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv

    def train_on_batch(self, X, y):
        X = transform_sentence_batch_to_vector(self.word_vectors, [lower_text(text) for text in X['body']],
                                               self.document_max_num_words, self.num_features)
        self.model.train_on_batch(X, y)

    def predict(self, X):
        X = transform_sentence_batch_to_vector(self.word_vectors, [lower_text(text) for text in X['body']],
                                               self.document_max_num_words, self.num_features)
        return np.round(self.model.predict(X))
