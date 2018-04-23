import os

from sklearn.model_selection import train_test_split

from fasttext.fasttext_wrapper import FasttextWrapper
from models.data_reader import DataReader
from models.utils import print_metrics


class FasttextClassifier:
    def __init__(self):
        self.batch_size = 50

    def process(self, csv_file_name):
        data_reader = DataReader(csv_file_name)
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        file_name = os.path.splitext(os.path.basename(csv_file_name))[0]
        train_file_name = 'fasttext/data/' + file_name + '_train.txt'
        test_file_name = 'fasttext/data/' + file_name + '_test.txt'
        fasttext = FasttextWrapper()
        fasttext.fit(train_file_name, data_reader, train_ids)

        y_preds = fasttext.predict(test_file_name, data_reader, test_ids)
        y_tests = []
        for X_test, y_test in data_reader.get_data_labels_batch(set(test_ids), self.batch_size):
            y_tests.extend(y_test)

        print_metrics(y_tests, y_preds)