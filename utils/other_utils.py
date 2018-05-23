import datetime
import logging
import os.path
import time


def get_dataset_name(csv_file_name):
    file_name = os.path.splitext(os.path.basename(csv_file_name))[0]
    if file_name.find('_') != -1:
        file_name = file_name[file_name.find('_') + 1:]
    return file_name


def get_word2vec_model_path(csv_file_name):
    return 'word2vec/word2vec_models/{}_model.bin'.format(get_dataset_name(csv_file_name))


def get_fasttext_model_path(csv_file_name):
    return 'word2vec/fasttext_models/{}_model.bin'.format(get_dataset_name(csv_file_name))


def get_logging_filename():
    return os.path.splitext(os.path.basename(logging.root.handlers[0].baseFilename))[0]


def string_to_timestamp(date):
    return int(time.mktime(datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f').timetuple()))