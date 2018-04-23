import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.data_reader import DataReader


class SKLearnClassifier:
    def __init__(self):
        self.batch_size = 50

    def process(self, csv_file_name, classifier, vectorizer):
        data_reader = DataReader(csv_file_name)
        ids = data_reader.get_ids()
        train_ids, test_ids = train_test_split(ids, random_state=0)
        vectorizer.fit(data_reader, train_ids)

        all_classes = np.array([0, 1])
        for X_train, y_train in data_reader.get_texts_labels_batch(set(train_ids), self.batch_size):
            X_train = vectorizer.transform(X_train)
            classifier.partial_fit(X_train, y_train, classes=all_classes)

        y_tests = []
        y_preds = []
        for X_test, y_test in data_reader.get_texts_labels_batch(set(test_ids), self.batch_size):
            X_test = vectorizer.transform(X_test)
            y_pred = classifier.predict(X_test)
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

        print('Accuracy: {}'.format(accuracy_score(y_tests, y_preds)))
        print('Precision: {}'.format(precision_score(y_tests, y_preds)))
        print('Recall: {}'.format(recall_score(y_tests, y_preds)))
        print('F1: {}'.format(f1_score(y_tests, y_preds)))
