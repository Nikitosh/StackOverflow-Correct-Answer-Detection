import gensim.downloader as api

from utils.utils import lower_text


class WikiSentences:
    MIN_SENTENCE_LENGTH = 10

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        data = api.load(self.filename)
        for article in data:
            for text in article['section_texts']:
                sentences = text.split('.')
                for sentence in sentences:
                    sentence = lower_text(sentence)
                    if len(sentence) >= WikiSentences.MIN_SENTENCE_LENGTH:
                        yield sentence.split()
