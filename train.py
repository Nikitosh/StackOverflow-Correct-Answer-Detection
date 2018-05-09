import logging

import numpy as np
from sklearn.model_selection import train_test_split

from data_reader.data_reader import DataReader
from utils.utils import print_metrics


def train(classifier, csv_file_name, batch_size=50, epochs=1):
    data_reader = DataReader(csv_file_name)
    ids = data_reader.get_ids()
    train_ids, test_ids = train_test_split(ids, random_state=3)

    classifier.pretrain(data_reader, train_ids)
    for epoch in range(epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
        losses = []
        accuracies = []
        for X_train, y_train in data_reader.get_raw_data_labels_batch(set(train_ids), batch_size):
            result = classifier.train_on_batch(X_train, y_train)
            losses.append(result[0])
            accuracies.append(result[1])
            logging.info('Loss: {}, accuracy: {}'.format(result[0], result[1]))
        logging.info('Mean loss for epoch {}: {}'.format(epoch + 1, np.mean(losses)))
        logging.info('Mean accuracy for epoch {}: {}'.format(epoch + 1, np.mean(accuracies)))

    y_tests = []
    y_preds = []
    for X_test, y_test in data_reader.get_raw_data_labels_batch(set(test_ids), batch_size):
        y_pred = classifier.predict(X_test)
        y_tests.extend(y_test)
        y_preds.extend(y_pred)

    print_metrics(y_tests, y_preds)

