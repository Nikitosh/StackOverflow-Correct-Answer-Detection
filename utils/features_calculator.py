from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup

from textstat.textstat_ext import textstat
from utils.utils import get_node_text, process_html, is_stop_word


class FeaturesCalculator:
    HTML_FEATURES_COUNT = 3
    TEXT_FEATURES_COUNT = 16
    OTHER_FEATURES_COUNT = 2
    QA_OVERLAP_FEATURES_COUNT = 4
    LINGUISTIC_FEATURES_COUNT = HTML_FEATURES_COUNT + TEXT_FEATURES_COUNT
    FEATURES_COUNT = LINGUISTIC_FEATURES_COUNT + OTHER_FEATURES_COUNT

    def __init__(self):
        self.calculated_features = {}
        self.word_count = defaultdict(int)
        self.documents = 0
        self.maximum_features = np.zeros(FeaturesCalculator.FEATURES_COUNT)
        self.answer_features_for_question = {}

    def handle_answer_question(self, answer_id, question_id, answer_html_text, age, score):
        answer_words = set(textstat.lexicon(process_html(answer_html_text)))
        for word in answer_words:
            self.word_count[word] += 1
        self.documents += 1
        features = self._calculate_features(answer_id, answer_html_text, age, score)
        self.maximum_features = np.maximum(self.maximum_features, features)
        if question_id not in self.answer_features_for_question:
            self.answer_features_for_question[question_id] = [[] for _ in range(len(features))]
        for i in range(len(features)):
            self.answer_features_for_question[question_id][i].append(features[i])

    def process_answers(self):
        for question_id in self.answer_features_for_question:
            for i in range(FeaturesCalculator.FEATURES_COUNT):
                self.answer_features_for_question[question_id][i].sort()
            self.answer_features_for_question[question_id][-2].reverse()

    def get_features(self, answer_id, question_id, answer_html_text, question_html_text):
        features = self.calculated_features[answer_id]
        answers = self.answer_features_for_question[question_id]
        normalized_features = np.divide(features, self.maximum_features)
        normalized_by_question_features = []
        features_relative_positions = []
        for i in range(FeaturesCalculator.FEATURES_COUNT):
            maximum_feature_value = max(answers[i]) if max(answers[i]) > 0 else 1
            normalized_by_question_features.append(features[i] / maximum_feature_value)
            features_relative_positions.append(answers[i].index(features[i]) / len(answers[i]))
        overlap_features = self._calculate_overlap_features(process_html(answer_html_text),
                                                            process_html(question_html_text))
        return normalized_features.tolist() + normalized_by_question_features + features_relative_positions + overlap_features

    def _calculate_features(self, id, html_text, age, score):
        if id not in self.calculated_features:
            soup = BeautifulSoup(html_text, 'html.parser')
            text = ' '.join(list(filter(None, [get_node_text(node).strip() for node in soup.childGenerator()])))
            self.calculated_features[id] = FeaturesCalculator.get_html_features(
                soup) + FeaturesCalculator.get_text_features(text) + [age, score]
        return self.calculated_features[id]

    @staticmethod
    def get_html_features(soup):
        return [FeaturesCalculator.get_tag_count(soup, 'a'),
                FeaturesCalculator.get_tag_count(soup, 'code'),
                FeaturesCalculator.get_tag_count(soup, 'p')]

    @staticmethod
    def get_tag_count(soup, tag):
        return len(soup.find_all(tag))

    @staticmethod
    def get_text_features(text):
        if textstat.lexicon_count(text) == 0:
            return [0 for _ in range(FeaturesCalculator.TEXT_FEATURES_COUNT)]
        length = len(text)
        return [
            textstat.uppercase_letter_count(text) / length,
            textstat.lowercase_letter_count(text) / length,
            textstat.space_count(text) / length,
            length,
            textstat.longest_sentence_char_count(text),
            textstat.longest_sentence_lexicon_count(text),
            textstat.lexicon_count(text) / textstat.sentence_count(text),
            textstat.letter_count(text) / textstat.lexicon_count(text),
            textstat.sentence_count(text),
            textstat.automated_readability_index(text),
            textstat.flesch_reading_ease(text),
            textstat.smog_index(text),
            textstat.flesch_kincaid_grade(text),
            textstat.coleman_liau_index(text),
            textstat.gunning_fog(text),
            textstat.lix(text)
        ]

    def _calculate_overlap_features(self, answer_text, question_text):
        answer_words = set(textstat.lexicon(answer_text))
        answer_filtered_words = set(filter(lambda word: not is_stop_word(word), answer_words))
        question_words = set(textstat.lexicon(question_text))
        question_filtered_words = set(filter(lambda word: not is_stop_word(word), question_words))
        features = []
        for a_words, q_words in [(answer_words, question_words), (answer_filtered_words, question_filtered_words)]:
            union = a_words.union(q_words)
            intersection = a_words.intersection(q_words)
            features.append(len(intersection) / len(union))
            intersection_idf = 0
            union_idf = 0
            for word in intersection:
                intersection_idf += np.log(self.documents / (self.word_count[word] + 1))
            for word in union:
                union_idf += np.log(self.documents / (self.word_count[word] + 1))
            features.append(intersection_idf / union_idf)
        return features