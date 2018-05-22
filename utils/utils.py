import datetime
import logging
import os.path
import re
import time

import gensim
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec, FastText
from keras.layers import Bidirectional, LSTM, Embedding
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

stops = set(stopwords.words('english'))

IGNORED_TAGS = ['del', 'strike', 's']
CODE_TAG = 'code'
THREAD_FEATURES_COUNT = 2


def lower_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)
    text = text.strip()
    return text


def transform_text(text):
    text = lower_text(text)
    filtered_words = [word for word in text.split() if is_stop_word(word) and len(word) >= 3]
    return ' '.join(filtered_words)


def stem_text(text):
    return gensim.parsing.preprocessing.stem_text(transform_text(text))


def is_stop_word(word):
    return word in stops


def print_metrics(epoch, y_tests, y_preds, threshold=0.5):
    y_preds_binary = list(map(lambda x: x >= threshold, y_preds))
    logging.info('Test accuracy for epoch #{}: {}'.format(epoch, accuracy_score(y_tests, y_preds_binary)))
    logging.info('Test precision for epoch #{}: {}'.format(epoch, precision_score(y_tests, y_preds_binary)))
    logging.info('Test recall for epoch #{}: {}'.format(epoch, recall_score(y_tests, y_preds_binary)))
    logging.info('Test F1 for epoch #{}: {}'.format(epoch, f1_score(y_tests, y_preds_binary)))
    draw_precision_recall_curve(y_tests, y_preds)
    draw_roc_curve(y_tests, y_preds)


def generate_unit_vector(dim):
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def get_embedding(data_reader, ids):
    #word_vectors = Word2Vec.load(get_word2vec_model_path(data_reader.csv_file_name)).wv
    model = FastText.load(get_fasttext_model_path(data_reader.csv_file_name))

    all_words = set()
    for words in data_reader.get_processed_questions_and_answers_as_lists(ids):
        all_words.update(words)

    words_count = len(all_words) + 1
    num_features = 300
    embedding_weights = np.zeros((words_count, num_features))
    words_index = {}
    for index, word in enumerate(all_words):
        words_index[word] = index + 1
        if word in model:
            embedding_weights[index + 1, :] = model[word]
        else:
            logging.info(word)

    embedding = Embedding(output_dim=num_features, input_dim=words_count, trainable=False)
    embedding.build((None,))
    embedding.set_weights([embedding_weights])
    return embedding, words_index


def get_dataset_name(csv_file_name):
    file_name = os.path.splitext(os.path.basename(csv_file_name))[0]
    if file_name.find('_') != -1:
        file_name = file_name[file_name.find('_') + 1:]
    return file_name


def get_word2vec_model_path(csv_file_name):
    return 'word2vec/word2vec_models/{}_model.bin'.format(get_dataset_name(csv_file_name))


def get_fasttext_model_path(csv_file_name):
    return 'word2vec/fasttext_models/{}_model.bin'.format(get_dataset_name(csv_file_name))


def get_lstm(embed_size, bidirectional, dropout, return_sequences=False):
    if bidirectional:
        return Bidirectional(
            LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
    else:
        return LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)


def string_to_timestamp(date):
    return int(time.mktime(datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f').timetuple()))


def get_logging_filename():
    return os.path.splitext(os.path.basename(logging.root.handlers[0].baseFilename))[0]


def process_code(code):
    return 'CODELEXEM'


def get_child_text(node):
    name = getattr(node, 'name', None)
    if name in IGNORED_TAGS:
        return ''
    if name == CODE_TAG:
        return process_code(get_node_text(node))
    return get_node_text(node)


def get_node_text(node):
    if 'childGenerator' in dir(node):
        return ' '.join([get_child_text(child) for child in node.childGenerator()])
    return '' if node.isspace() else node


def process_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return ' '.join(list(filter(None, [get_node_text(node).strip() for node in soup.childGenerator()])))


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
