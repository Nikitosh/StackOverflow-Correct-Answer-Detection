import logging

import numpy as np
from bs4 import BeautifulSoup
from textstat.textstat import textstat

from utils.utils import get_node_text


class LinguisticFeaturesCalculator():
    HTML_FEATURES_COUNT = 3
    TEXT_FEATURES_COUNT = 11
    LINGUISTIC_FEATURES_COUNT = HTML_FEATURES_COUNT + TEXT_FEATURES_COUNT

    maximum_features = np.zeros(LINGUISTIC_FEATURES_COUNT)
    calculated_features = {}

    def precalculate_maximum_linguistic_features(self, data_reader, ids):
        for X, y in data_reader.get_raw_data_labels_batch(ids, batch_size=50):
            for id, text in zip(X.id, X.body):
                self.maximum_features = np.maximum(self.maximum_features, self.get_linguistic_features(id, text))

    def get_normalized_linguistic_features(self, id, html_text):
        return self.get_linguistic_features(id, html_text) / self.maximum_features

    def get_linguistic_features(self, id, html_text):
        if id % 1000 == 0:
            logging.info(str(id))
        if id not in self.calculated_features:
            soup = BeautifulSoup(html_text, 'html.parser')
            text = ' '.join(list(filter(None, [get_node_text(node).strip() for node in soup.childGenerator()])))
            text_features = [0 for _ in range(LinguisticFeaturesCalculator.TEXT_FEATURES_COUNT)]
            if textstat.sentence_count(text) != 0 and textstat.lexicon_count(text) != 0:
                text_features = LinguisticFeaturesCalculator.get_text_features(text)
            self.calculated_features[id] = LinguisticFeaturesCalculator.get_html_features(soup) + text_features
        return self.calculated_features[id]

    @staticmethod
    def get_html_features(soup):
        return [LinguisticFeaturesCalculator.get_href_tags_count(soup),
                LinguisticFeaturesCalculator.get_code_tags_count(soup),
                LinguisticFeaturesCalculator.get_p_tags_count(soup)]

    @staticmethod
    def get_text_features(text):
        return [LinguisticFeaturesCalculator.get_length(text),
                LinguisticFeaturesCalculator.get_lowercase_percentage(text),
                LinguisticFeaturesCalculator.get_uppercase_percentage(text),
                LinguisticFeaturesCalculator.get_spaces_percentage(text)] \
               + LinguisticFeaturesCalculator.get_indices(text)

    @staticmethod
    def get_href_tags_count(soup):
        return len(soup.find_all('a'))

    @staticmethod
    def get_code_tags_count(soup):
        return len(soup.find_all('code'))

    @staticmethod
    def get_p_tags_count(soup):
        return len(soup.find_all('p'))

    @staticmethod
    def get_length(text):
        return len(text)

    @staticmethod
    def get_lowercase_percentage(text):
        return sum(1 for c in text if c.islower()) / len(text)

    @staticmethod
    def get_uppercase_percentage(text):
        return sum(1 for c in text if c.isupper()) / len(text)

    @staticmethod
    def get_spaces_percentage(text):
        return sum(1 for c in text if c == ' ') / len(text)

    @staticmethod
    def get_indices(text):
        return [
            textstat.automated_readability_index(text),
            textstat.flesch_reading_ease(text),
            textstat.smog_index(text),
            textstat.flesch_kincaid_grade(text),
            textstat.coleman_liau_index(text),
            textstat.gunning_fog(text),
            textstat.lix(text)
        ]
