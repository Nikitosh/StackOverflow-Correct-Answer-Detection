import logging

from keras import Sequential
from keras.layers import Dropout, Dense, Merge

from gensim.models.word2vec import Word2Vec
from utils.utils import transform_sentence_batch_to_vector, lower_text, get_word2vec_model_path, add_lstm_to_model, \
    process_html
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class RnnWord2VecWithQuestionClassifier:
    def __init__(self, question_title_words_count, question_body_words_count, answer_body_words_count, lstm_embed_size,
                 bidirectional, dropout, mode):
        self.question_title_words_count = question_title_words_count
        self.question_body_words_count = question_body_words_count
        self.answer_body_words_count = answer_body_words_count
        self.num_features = 300

        self.word_vectors = None

        self.question_title_model = Sequential()
        question_title_input_shape = (self.question_title_words_count, self.num_features)
        add_lstm_to_model(self.question_title_model, lstm_embed_size, bidirectional, question_title_input_shape,
                          dropout)
        self.question_title_model.add(Dropout(dropout))

        self.question_body_model = Sequential()
        question_body_input_shape = (self.question_body_words_count, self.num_features)
        add_lstm_to_model(self.question_body_model, lstm_embed_size, bidirectional, question_body_input_shape, dropout)
        self.question_body_model.add(Dropout(dropout))

        self.answer_body_model = Sequential()
        answer_body_input_shape = (self.answer_body_words_count, self.num_features)
        add_lstm_to_model(self.answer_body_model, lstm_embed_size, bidirectional, answer_body_input_shape, dropout)
        self.answer_body_model.add(Dropout(dropout))

        self.model = Sequential()
        self.model.add(Merge([self.question_title_model, self.question_body_model, self.answer_body_model], mode=mode))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.question_title_model.summary(print_fn=logging.info)
        self.question_body_model.summary(print_fn=logging.info)
        self.answer_body_model.summary(print_fn=logging.info)
        self.model.summary(print_fn=logging.info)

    def get_X(self, word_vectors, data):
        question_title = transform_sentence_batch_to_vector(word_vectors,
                                                            [lower_text(process_html(X)) for X in data['question_title']],
                                                            self.question_title_words_count, self.num_features)
        question_body = transform_sentence_batch_to_vector(word_vectors,
                                                           [lower_text(process_html(X)) for X in data['question_body']],
                                                           self.question_body_words_count, self.num_features)
        answer_body = transform_sentence_batch_to_vector(word_vectors,
                                                         [lower_text(process_html(X)) for X in data['body']],
                                                         self.answer_body_words_count, self.num_features)
        return [question_title, question_body, answer_body]

    def pretrain(self, data_reader, ids):
        trainer = Word2VecModelTrainer()
        trainer.train(data_reader, ids)
        self.word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv

    def train_on_batch(self, X, y):
        return self.model.train_on_batch(self.get_X(self.word_vectors, X), y)

    def predict(self, X):
        return self.model.predict(self.get_X(self.word_vectors, X))
