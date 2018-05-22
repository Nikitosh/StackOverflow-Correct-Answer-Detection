import logging

import numpy as np
from sklearn.model_selection import train_test_split

from utils.imbalanced_data_reader import ImbalancedDataReader
from utils.utils import print_metrics, draw_accuracy_curve, draw_loss_curve


def run(classifier, csv_file_name, batch_size=64, epochs=1):
    data_reader = ImbalancedDataReader(csv_file_name, 'question_id')
    ids = list(data_reader.get_ids())
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=2)
    train_ids, validation_ids = train_test_split(train_ids, test_size=0.2, random_state=2)

    classifier.pretrain(data_reader, ids)
    train_losses_per_epoch = []
    train_accuracies_per_epoch = []
    validation_losses_per_epoch = []
    validation_accuracies_per_epoch = []

    for epoch in range(1, epochs + 1):
        logging.info('Epoch #{}/{}'.format(epoch, epochs))

        loss, accuracy = train(epoch, classifier, data_reader, train_ids, batch_size)
        train_losses_per_epoch.append(loss)
        train_accuracies_per_epoch.append(accuracy)

        loss, accuracy = validate(epoch, classifier, data_reader, validation_ids, batch_size)
        validation_losses_per_epoch.append(loss)
        validation_accuracies_per_epoch.append(accuracy)

        test(epoch, classifier, data_reader, test_ids, batch_size)
        classifier.save(epoch)

    draw_accuracy_curve(train_accuracies_per_epoch, validation_accuracies_per_epoch)
    draw_loss_curve(train_losses_per_epoch, validation_losses_per_epoch)


def train(epoch, classifier, data_reader, train_ids, batch_size):
    losses = []
    accuracies = []
    train_size = 2 * len(train_ids)

    batch_index = 0
    for X_train, y_train in data_reader.get_raw_data_labels_batch(set(train_ids), batch_size):
        result = classifier.train_on_batch(X_train, y_train)
        losses.append(result[0])
        accuracies.append(result[1])
        batch_index += 1
        logging.info(
            'Training batch #{}/{}: loss: {}, accuracy: {}'.format(batch_index, train_size // batch_size, result[0],
                                                                   result[1]))
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    logging.info('Mean training loss for epoch #{}: {}'.format(epoch, loss))
    logging.info('Mean training accuracy for epoch #{}: {}'.format(epoch, accuracy))
    return loss, accuracy


def validate(epoch, classifier, data_reader, validation_ids, batch_size):
    losses = []
    accuracies = []
    validation_size = 2 * len(validation_ids)

    batch_index = 0
    for X_validation, y_validation in data_reader.get_raw_data_labels_batch(set(validation_ids), batch_size):
        result = classifier.evaluate(X_validation, y_validation)
        losses.append(result[0])
        accuracies.append(result[1])
        batch_index += 1
        logging.info(
            'Validation batch #{}/{}: loss: {}, accuracy: {}'.format(batch_index, validation_size // batch_size, result[0],
                                                                     result[1]))
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    logging.info('Mean validation loss for epoch #{}: {}'.format(epoch, loss))
    logging.info('Mean validation accuracy for epoch #{}: {}'.format(epoch, accuracy))
    return loss, accuracy


def test(epoch, classifier, data_reader, test_ids, batch_size):
    y_tests = []
    y_preds = []
    test_size = 2 * len(test_ids)

    batch_index = 0
    for X_test, y_test in data_reader.get_raw_data_labels_batch(set(test_ids), batch_size):
        y_pred = classifier.predict(X_test)
        y_tests.extend(y_test)
        y_preds.extend(y_pred)
        batch_index += 1
        logging.info('Test batch #{}/{}'.format(batch_index, test_size // batch_size))
    print_metrics(epoch, y_tests, y_preds)
