import logging

from gensim.models.word2vec import Word2Vec
from keras.utils import plot_model

from utils.linguistic_features_calculator import LinguisticFeaturesCalculator
from utils.utils import get_word2vec_model_path, get_logging_filename
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


class NeuralNetClassifier:
    def __init__(self):
        self.num_features = 300
        self.model = None
        self.word_vectors = None
        self.linguistic_features_calculator = LinguisticFeaturesCalculator()

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary(print_fn=logging.info)
        plot_model(self.model, to_file='outputs/models_images/model-{}.png'.format(get_logging_filename()))

    def pretrain(self, data_reader, ids):
        trainer = Word2VecModelTrainer()
        trainer.train(data_reader, ids)
        self.word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv
        self.linguistic_features_calculator.precalculate_maximum_linguistic_features(data_reader, ids)

    def transform_input(self, X):
        raise NotImplementedError

    def train_on_batch(self, X, y):
        return self.model.train_on_batch(self.transform_input(X), y)

    def evaluate(self, X, y):
        return self.model.evaluate(self.transform_input(X), y)

    def predict(self, X):
        return self.model.predict(self.transform_input(X))

    def save(self, epoch):
        self.model.save('outputs/models/model-{}-{}.h5'.format(get_logging_filename(), epoch))
