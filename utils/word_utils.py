import logging
import re

import gensim
import numpy as np
from gensim.models import FastText
from keras.layers import Embedding
from nltk.corpus import stopwords

from utils.other_utils import get_fasttext_model_path

stops = set(stopwords.words('english'))


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


def get_embedding(data_reader, ids):
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