import re

import gensim
from nltk.corpus import stopwords

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
