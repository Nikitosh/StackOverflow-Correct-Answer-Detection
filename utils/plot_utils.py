import logging

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, \
    precision_score, recall_score, f1_score

from utils.other_utils import get_logging_filename


def draw_roc_curve(y_tests, y_preds):
    fpr, tpr, _ = roc_curve(y_tests, y_preds)
    logging.info('AUC: {}'.format(auc(fpr, tpr)))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig('outputs/plots/roc-{}.png'.format(get_logging_filename()))


def draw_precision_recall_curve(y_tests, y_preds):
    precision, recall, _ = precision_recall_curve(y_tests, y_preds)
    average_precision = average_precision_score(y_tests, y_preds)
    logging.info('Average precision-recall score: {0:0.2f}'.format(average_precision))

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('outputs/plots/precision_recall-{}.png'.format(get_logging_filename()))


def draw_accuracy_curve(train_accuracies, validation_accuracies):
    plt.figure()
    plt.plot(train_accuracies)
    plt.plot(validation_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('outputs/plots/accuracy-{}.png'.format(get_logging_filename()))


def draw_loss_curve(train_losses, validation_losses):
    plt.figure()
    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('outputs/plots/loss-{}.png'.format(get_logging_filename()))


def print_metrics(epoch, y_tests, y_preds, threshold=0.5):
    y_preds_binary = list(map(lambda x: x >= threshold, y_preds))
    logging.info('Test accuracy for epoch #{}: {}'.format(epoch, accuracy_score(y_tests, y_preds_binary)))
    logging.info('Test precision for epoch #{}: {}'.format(epoch, precision_score(y_tests, y_preds_binary)))
    logging.info('Test recall for epoch #{}: {}'.format(epoch, recall_score(y_tests, y_preds_binary)))
    logging.info('Test F1 for epoch #{}: {}'.format(epoch, f1_score(y_tests, y_preds_binary)))
    draw_precision_recall_curve(y_tests, y_preds)
    draw_roc_curve(y_tests, y_preds)