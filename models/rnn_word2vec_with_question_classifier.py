import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional, Merge

from sklearn.model_selection import train_test_split

from gensim.models.word2vec import Word2Vec
from models.data_reader import DataReader
from models.utils import transform_sentence_batch_to_vector, print_metrics, transform_text, lower_text, \
    get_word2vec_model_path


class RnnWord2VecWithQuestionClassifier:
    def __init__(self):
        self.batch_size = 50
        self.num_features = 300
        self.question_title_max_num_words = 50
        self.question_body_max_num_words = 500
        self.answer_body_max_num_words = 500
        self.embed_size = 100

        self.question_title_model = Sequential()
        self.question_title_model.add(
            Bidirectional(LSTM(self.embed_size), input_shape=(self.question_title_max_num_words, self.num_features)))
        self.question_title_model.add(Dropout(0.2))

        self.question_body_model = Sequential()
        self.question_body_model.add(
            Bidirectional(LSTM(self.embed_size), input_shape=(self.question_body_max_num_words, self.num_features)))
        self.question_body_model.add(Dropout(0.2))

        self.answer_body_model = Sequential()
        self.answer_body_model.add(
            Bidirectional(LSTM(self.embed_size), input_shape=(self.answer_body_max_num_words, self.num_features)))
        self.answer_body_model.add(Dropout(0.2))

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

    def process(self, csv_file_name):
        data_reader = DataReader(csv_file_name)

        word_vectors = Word2Vec.load(get_word2vec_model_path(csv_file_name)).wv
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        for X_train, y_train in data_reader.get_data_labels_batch(set(train_ids), self.batch_size):
            self.model.train_on_batch(self.get_X(word_vectors, X_train), y_train)

        y_tests = []
        y_preds = []
        for X_test, y_test in data_reader.get_data_labels_batch(set(test_ids), self.batch_size):
            y_pred = np.round(self.model.predict(self.get_X(word_vectors, X_test)))
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

        print_metrics(y_tests, y_preds)
