import os

from fasttext.fasttext_wrapper import FasttextWrapper


class FasttextClassifier:
    def __init__(self):
        self.fasttext = None

    def pretrain(self, data_reader, ids):
        file_name = os.path.splitext(os.path.basename(data_reader.csv_file_name))[0]
        train_file_name = 'fasttext/data/' + file_name + '_train.txt'
        self.fasttext = FasttextWrapper()
        self.fasttext.fit(train_file_name, data_reader, ids)

    def train_on_batch(self, X, y):
        pass

    def predict(self, X):
        pass