import numpy as np
from keras import Sequential
from keras.layers import Dropout, Dense, Merge

from gensim.models.word2vec import Word2Vec
from models.utils import transform_sentence_batch_to_vector, lower_text, get_word2vec_model_path, add_lstm_to_model
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class RnnWord2VecWithQuestionClassifier:
    def __init__(self, bidirectional=False, dropout=0.1):
        self.batch_size = 50
        self.num_features = 300
        self.question_title_max_num_words = 50
        self.question_body_max_num_words = 500
        self.answer_body_max_num_words = 500
        self.embed_size = 100

        self.word_vectors = None

        self.question_title_model = Sequential()
        question_title_input_shape = (self.question_title_max_num_words, self.num_features)
        add_lstm_to_model(self.question_title_model, self.embed_size, bidirectional, question_title_input_shape, dropout)
        self.question_title_model.add(Dropout(dropout))

        self.question_body_model = Sequential()
        question_body_input_shape = (self.question_body_max_num_words, self.num_features)
        add_lstm_to_model(self.question_body_model, self.embed_size, bidirectional, question_body_input_shape, dropout)
        self.question_body_model.add(Dropout(dropout))

        self.answer_body_model = Sequential()
        answer_body_input_shape = (self.answer_body_max_num_words, self.num_features)
        add_lstm_to_model(self.answer_body_model, self.embed_size, bidirectional, answer_body_input_shape, dropout)
        self.answer_body_model.add(Dropout(dropout))

        self.model = Sequential()
        self.model.add(Merge([self.question_title_model, self.question_body_model, self.answer_body_model], mode='concat'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.question_title_model.summary())
        print(self.question_body_model.summary())
        print(self.answer_body_model.summary())
        print(self.model.summary())

    def get_X(self, word_vectors, data):
        question_title = transform_sentence_batch_to_vector(word_vectors,
                                                            [lower_text(X) for X in data['question_title']],
                                                            self.question_title_max_num_words, self.num_features)
        question_body = transform_sentence_batch_to_vector(word_vectors,
                                                           [lower_text(X) for X in data['question_body']],
                                                           self.question_body_max_num_words, self.num_features)
        answer_body = transform_sentence_batch_to_vector(word_vectors,
                                                         [lower_text(X) for X in data['body']],
                                                         self.answer_body_max_num_words, self.num_features)
        return [question_title, question_body, answer_body]

    def pretrain(self, data_reader, ids):
        trainer = Word2VecModelTrainer()
        trainer.train(data_reader, ids)
        self.word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv

    def train_on_batch(self, X, y):
        self.model.train_on_batch(self.get_X(self.word_vectors, X), y)

    def predict(self, X):
        return np.round(self.model.predict(self.get_X(self.word_vectors, X)))
