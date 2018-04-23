import numpy as np

from sklearn.model_selection import train_test_split

from models.data_reader import DataReader
from models.utils import print_metrics, transform_text, stem_text


class SKLearnClassifier:
    def __init__(self):
        self.batch_size = 50

    def process(self, csv_file_name, classifier, vectorizer):
        data_reader = DataReader(csv_file_name)
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        vectorizer.fit(data_reader, train_ids)

        all_classes = np.array([0, 1])
        for X_train, y_train in data_reader.get_data_labels_batch(set(train_ids), self.batch_size):
            X_train = vectorizer.transform(stem_text(X) for X in X_train['body'])
            classifier.partial_fit(X_train, y_train, classes=all_classes)

        y_tests = []
        y_preds = []
        for X_test, y_test in data_reader.get_data_labels_batch(set(test_ids), self.batch_size):
            X_test = vectorizer.transform(stem_text(X) for X in X_test['body'])
            y_pred = classifier.predict(X_test)
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

        print_metrics(y_tests, y_preds)
