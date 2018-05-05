import numpy as np
from gensim.models import KeyedVectors
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional

from sklearn.model_selection import train_test_split

from gensim.models.word2vec import Word2Vec
from models.data_reader import DataReader
from models.utils import transform_sentence_batch_to_vector, print_metrics, transform_text, lower_text, \
    get_word2vec_model_path
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class RnnWord2VecClassifier:
    def __init__(self):
        self.batch_size = 50
        self.num_features = 300
        self.document_max_num_words = 300

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(100), input_shape=(self.document_max_num_words, self.num_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def process(self, csv_file_name):
        data_reader = DataReader(csv_file_name)

        word_vectors = Word2Vec.load(get_word2vec_model_path(csv_file_name)).wv
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        for X_train, y_train in data_reader.get_data_labels_batch(set(train_ids), self.batch_size):
            X = transform_sentence_batch_to_vector(word_vectors, [lower_text(X) for X in X_train['body']],
                                                   self.document_max_num_words, self.num_features)
            self.model.train_on_batch(X, y_train)

        y_tests = []
        y_preds = []
        for X_test, y_test in data_reader.get_data_labels_batch(set(test_ids), self.batch_size):
            X = transform_sentence_batch_to_vector(word_vectors, [lower_text(X) for X in X_test['body']],
                                                   self.document_max_num_words, self.num_features)
            y_pred = np.round(self.model.predict(X))
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

        print_metrics(y_tests, y_preds)
