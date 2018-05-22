import logging

import numpy as np
from keras.utils import plot_model

from utils.features_calculator import FeaturesCalculator
from utils.utils import get_logging_filename, get_embedding
from word2vec.fasttext_model_trainer import FasttextModelTrainer
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class NeuralNetClassifier:
    def __init__(self):
        self.num_features = 300
        self.model = None
        self.embedding = None
        self.words_index = None
        self.linguistic_features_calculator = FeaturesCalculator()
        self.class_weight = {0: 0.5, 1: 0.5}

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary(print_fn=logging.info)
        plot_model(self.model, to_file='outputs/models_images/model-{}.png'.format(get_logging_filename()))

    def pretrain(self, data_reader, ids):
        trainer = FasttextModelTrainer()
        trainer.train(data_reader, ids)
        self.embedding, self.words_index = get_embedding(data_reader, ids)
        self.linguistic_features_calculator.precalculate_maximum_linguistic_features(data_reader, ids)
        labels_number = [0, 0]
        for X, y in data_reader.get_raw_data_labels_batch(ids, batch_size=64):
            for y_i in y:
                labels_number[y_i] += 1
        all_labels_number = labels_number[0] + labels_number[1]
        self.class_weight = {0: labels_number[1] / all_labels_number, 1: labels_number[0] / all_labels_number}
        logging.info('Labels number: {}'.format(labels_number))
        self.create_model()

    def create_model(self):
        raise NotImplementedError

    def transform_input(self, X):
        raise NotImplementedError

    def train_on_batch(self, X, y):
        return self.model.train_on_batch(self.transform_input(X), y, class_weight=self.class_weight)

    def evaluate(self, X, y):
        return self.model.evaluate(self.transform_input(X), y)

    def predict(self, X):
        return self.model.predict(self.transform_input(X))

    def save(self, epoch):
        self.model.save('outputs/models/model-{}-{}.h5'.format(get_logging_filename(), epoch))

    def transform_sentence_batch_to_vector(self, sentences, document_max_num_words):
        X = np.zeros((len(sentences), document_max_num_words))
        for i in range(len(sentences)):
            words = sentences[i].split()
            for j, word in enumerate(words):
                if j == document_max_num_words:
                    break
                if word in self.words_index:
                    X[i, j] = self.words_index[word]
                else:
                    logging.info('Unknown word: {}'.format(word))
        return X
