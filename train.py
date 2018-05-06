import numpy as np
from sklearn.model_selection import train_test_split

from models.data_reader import DataReader
from models.utils import print_metrics


def train(classifier, csv_file_name, batch_size=50):
    data_reader = DataReader(csv_file_name)
    ids = data_reader.get_ids()
    train_ids, test_ids = train_test_split(ids, random_state=1)

    classifier.pretrain(data_reader, train_ids)
    for X_train, y_train in data_reader.get_data_labels_batch(set(train_ids), batch_size):
        classifier.train_on_batch(X_train, y_train)

    y_tests = []
    y_preds = []
    for X_test, y_test in data_reader.get_data_labels_batch(set(test_ids), batch_size):
        y_pred = np.round(classifier.predict(X_test))
        y_tests.extend(y_test)
        y_preds.extend(y_pred)

    print_metrics(y_tests, y_preds)

