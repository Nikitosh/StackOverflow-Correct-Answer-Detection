import csv
import os
import re

from lxml import etree

from utils.features_calculator import FeaturesCalculator
from utils.utils import string_to_timestamp


class XmlToCsvPreprocessor:
    QUESTION_ID = 1
    ANSWER_ID = 2
    QUESTION_SCORE_THRESHOLD = 1

    QUESTION_WORDS = frozenset([
        'how',
        'what',
        'why',
        'which',
        'where',
        'when',
        'who',
        'is',
        'are',
        'can',
        'could',
        'will',
        'would',
        'should',
        'do',
        'does',
        'did',
        'have',
        'has',
        'had'
    ])

    @staticmethod
    def iterate_xml(xml_file):
        doc = etree.iterparse(xml_file, events=('start', 'end'))
        _, root = next(doc)
        start_tag = None
        for event, element in doc:
            if event == 'start' and start_tag is None:
                start_tag = element.tag
            if event == 'end' and element.tag == start_tag:
                yield element
                start_tag = None
                root.clear()

    @staticmethod
    def is_question(sentence):
        return sentence[-1] == '?' and re.split(' |\'|,', sentence)[0].lower() in XmlToCsvPreprocessor.QUESTION_WORDS

    def process_text(self, text):
        return text.replace('\n', ' ').strip()

    def process(self, xml_file_name):
        file_name = os.path.splitext(xml_file_name)[0]

        accepted_answer_ids = set()
        questions_with_accepted_answer_ids = set()
        questions = {}
        features_calculator = FeaturesCalculator()

        for elem in self.iterate_xml(xml_file_name):
            id = int(elem.get('Id'))
            type_id = int(elem.get('PostTypeId'))
            score = int(elem.get('Score'))
            date = string_to_timestamp(elem.get('CreationDate'))
            if type_id == XmlToCsvPreprocessor.QUESTION_ID and elem.get('AcceptedAnswerId') is not None \
                    and score >= XmlToCsvPreprocessor.QUESTION_SCORE_THRESHOLD:
                accepted_answer_ids.add(int(elem.get('AcceptedAnswerId')))
                questions_with_accepted_answer_ids.add(id)
                answer_count = int(elem.get('AnswerCount'))
                questions[id] = [id, date, score, self.process_text(elem.get('Title')),
                                 self.process_text(elem.get('Body')), answer_count]
            elif type_id == XmlToCsvPreprocessor.ANSWER_ID:
                parent_id = int(elem.get('ParentId'))
                body = self.process_text(elem.get('Body'))
                if parent_id not in questions_with_accepted_answer_ids or len(body) == 0:
                    continue
                features_calculator.handle_answer_question(id, questions[parent_id][0], self.process_text(body),
                                                           date - questions[parent_id][1], score)

        features_calculator.process_answers()

        with open(file_name + '.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(
                ['id', 'body',
                 'question_id', 'question_date', 'question_score', 'question_title', 'question_body', 'answer_count',

                 'a_count_n1', 'code_count_n1', 'p_count_n1', 'upper_count_n1', 'lower_count_n1', 'space_count_n1',
                 'length_n1', 'longest_sentence_char_count_n1', 'longest_sentence_word_count_n1', 'average_words_n1',
                 'average_chars_n1', 'sentence_count_n1', 'ari_n1', 'fre_n1', 'si_n1', 'fkg_n1', 'cli_n1', 'gf_n1',
                 'lix_n1', 'age_n1', 'score_n1',

                 'a_count_n2', 'code_count_n2', 'p_count_n2', 'upper_count_n2', 'lower_count_n2', 'space_count_n2',
                 'length_n2', 'longest_sentence_char_count_n2', 'longest_sentence_word_count_n2', 'average_words_n2',
                 'average_chars_n2', 'sentence_count_n2', 'ari_n2', 'fre_n2', 'si_n2', 'fkg_n2', 'cli_n2', 'gf_n2',
                 'lix_n2', 'age_n2', 'score_n2',

                 'a_count_pos', 'code_count_pos', 'p_count_pos', 'upper_count_pos', 'lower_count_pos', 'space_count_pos',
                 'length_pos', 'longest_sentence_char_count_pos', 'longest_sentence_word_count_pos', 'average_words_pos',
                 'average_chars_pos', 'sentence_count_pos', 'ari_pos', 'fre_pos', 'si_pos', 'fkg_pos', 'cli_pos', 'gf_pos',
                 'lix_pos', 'age_pos', 'score_pos',

                 'qa_overlap', 'qa_idf_overlap', 'qa_filtered_overlap', 'qa_filtered_idf_overlap',
                 'is_accepted'])
            for elem in self.iterate_xml(xml_file_name):
                id = int(elem.get('Id'))
                type_id = int(elem.get('PostTypeId'))
                score = int(elem.get('Score'))
                date = elem.get('CreationDate')

                if type_id == XmlToCsvPreprocessor.ANSWER_ID:
                    body = self.process_text(elem.get('Body'))
                    parent_id = int(elem.get('ParentId'))
                    if parent_id not in questions_with_accepted_answer_ids or len(body) == 0:
                        continue
                    is_accepted = int(id in accepted_answer_ids)
                    features = features_calculator.get_features(id, parent_id, body, questions[parent_id][-2])
                    writer.writerow([id, body]
                                    + questions[parent_id]
                                    + features
                                    + [is_accepted])
