import os
import re
import csv
import argparse
from collections import defaultdict

from lxml import etree
from bs4 import BeautifulSoup

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
        answers_times_for_question = defaultdict(list)
        maximum_answer_score_for_question = {}

        for elem in self.iterate_xml(xml_file_name):
            id = int(elem.get('Id'))
            type_id = int(elem.get('PostTypeId'))
            score = int(elem.get('Score'))
            date = elem.get('CreationDate')
            if type_id == XmlToCsvPreprocessor.QUESTION_ID and elem.get('AcceptedAnswerId') is not None \
                    and score >= XmlToCsvPreprocessor.QUESTION_SCORE_THRESHOLD:
                accepted_answer_ids.add(int(elem.get('AcceptedAnswerId')))
                questions_with_accepted_answer_ids.add(id)
                questions[id] = [id, date, score, self.process_text(elem.get('Title')),
                                 self.process_text(elem.get('Body'))]
            elif type_id == XmlToCsvPreprocessor.ANSWER_ID:
                parent_id = int(elem.get('ParentId'))
                answers_times_for_question[parent_id].append(string_to_timestamp(date))
                if parent_id not in maximum_answer_score_for_question \
                        or maximum_answer_score_for_question[parent_id] < score:
                    maximum_answer_score_for_question[parent_id] = score

        for id in questions_with_accepted_answer_ids:
            answers_times_for_question[id].sort()

        with open(file_name + '.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(
                ['id', 'creation_date', 'score', 'relative_score', 'position', 'relative_position', 'body',
                 'question_id', 'question_creation_date', 'question_score', 'question_title', 'question_body',
                 'answers_count', 'is_accepted'])
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
                    relative_score = score / max(1, maximum_answer_score_for_question[parent_id])
                    position = answers_times_for_question[parent_id].index(string_to_timestamp(date)) + 1
                    relative_position = 1 - position / len(answers_times_for_question[parent_id])
                    writer.writerow([id, date, score, relative_score, position, relative_position, body]
                                    + questions[parent_id]
                                    + [len(answers_times_for_question[parent_id]), is_accepted])
