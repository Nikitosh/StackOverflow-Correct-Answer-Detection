import re
import gensim

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    filtered_words = [word for word in text.split() if word not in stops]
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    text = ' '.join(filtered_words)
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    return gensim.parsing.preprocessing.stem_text(text)

