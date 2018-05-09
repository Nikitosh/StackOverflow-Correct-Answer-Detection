import logging

from keras import Sequential
from keras.layers import Dropout, Dense

from gensim.models.word2vec import Word2Vec
from utils.utils import transform_sentence_batch_to_vector, lower_text, get_word2vec_model_path, add_lstm_to_model, \
    process_html
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class RnnWord2VecClassifier:
    def __init__(self, answer_body_words_count, lstm_embed_size, bidirectional, dropout):
        self.answer_body_words_count = answer_body_words_count
        self.num_features = 300

        self.word_vectors = None

        self.model = Sequential()
        input_shape = (self.answer_body_words_count, self.num_features)
        add_lstm_to_model(self.model, input_shape, lstm_embed_size, bidirectional, dropout)
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.summary(print_fn=logging.info)

    def pretrain(self, data_reader, ids):
        trainer = Word2VecModelTrainer()
        trainer.train(data_reader, ids)
        self.word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv

    def train_on_batch(self, X, y):
        X = transform_sentence_batch_to_vector(self.word_vectors,
                                               [lower_text(process_html(text)) for text in X['body']],
                                               self.answer_body_words_count, self.num_features)
        return self.model.train_on_batch(X, y)

    def predict(self, X):
        X = transform_sentence_batch_to_vector(self.word_vectors,
                                               [lower_text(process_html(text)) for text in X['body']],
                                               self.answer_body_words_count, self.num_features)
        return self.model.predict(X)
