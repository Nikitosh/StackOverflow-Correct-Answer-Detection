import os
import re
import csv
import argparse

from lxml import etree
from bs4 import BeautifulSoup


class Preprocessor:
    QUESTION_ID = 1
    ANSWER_ID = 2

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

    IGNORED_TAGS = ['del', 'strike', 's']
    CODE_TAG = 'code'

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
        return sentence[-1] == '?' and re.split(' |\'|,', sentence)[0].lower() in Preprocessor.QUESTION_WORDS

    @staticmethod
    def process_code(code):
        return 'CODE_LEXEM.'

    def get_child_text(self, node):
        name = getattr(node, 'name', None)
        if name in Preprocessor.IGNORED_TAGS:
            return ''
        if name == Preprocessor.CODE_TAG:
            return self.process_code(self.get_node_text(node))
        return self.get_node_text(node).replace('"', '')

    def get_node_text(self, node):
        if 'childGenerator' in dir(node):
            return ''.join([self.get_child_text(child) for child in node.childGenerator()])
        return '' if node.isspace() else node

    def process_html_body(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        return ' '.join(list(filter(None, [self.get_node_text(node).strip() for node in soup.childGenerator()])))

    def process(self, xml_file_name):
        file_name = os.path.splitext(xml_file_name)[0]

        accepted_answer_ids = set()
        questions = {}
        with open(file_name + '.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['id', 'creation_date', 'score', 'body', 'question_id', 'question_creation_date',
                             'question_score', 'question_title', 'question_body', 'is_accepted'])
            for elem in self.iterate_xml(xml_file_name):
                id = int(elem.get('Id'))
                type_id = int(elem.get('PostTypeId'))
                score = int(elem.get('Score'))
                date = elem.get('CreationDate')
                body = self.process_html_body(elem.get('Body'))

                if type_id == Preprocessor.QUESTION_ID:
                    questions[id] = [id, date, score, elem.get('Title').replace('"', ''), body]
                    if elem.get('AcceptedAnswerId') is not None:
                        accepted_answer_ids.add(int(elem.get('AcceptedAnswerId')))

                if type_id == Preprocessor.ANSWER_ID:
                    parent_id = int(elem.get('ParentId'))
                    if parent_id not in questions or len(body) == 0:
                        continue
                    is_accepted = int(id in accepted_answer_ids)
                    writer.writerow([id, date, score, body] + questions[parent_id] + [is_accepted])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', type=str, help='Path to .xml file')
    args = parser.parse_args()
    preprocessor = Preprocessor()
    preprocessor.process(args.xml_path)
