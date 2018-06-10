import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#mpl.rcParams.update({'font.size': 16})

from utils.plot_utils import draw_lens_histogram, draw_histogram

from utils.html_utils import process_html
from utils.imbalanced_data_reader import ImbalancedDataReader
from utils.word_utils import lower_text


def test_lengths(csv_path):
    data_reader = ImbalancedDataReader(csv_path, 'question_id')
    question_title_lengths = []
    question_body_lengths = []
    answer_body_lengths = []
    cnt = 0
    for X, y in data_reader.get_raw_data_labels_batch(data_reader.get_ids(), 50):
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
        for i in range(50):
            question_title_lengths.append(len(lower_text(process_html(X.iloc[i]['question_title']))))
            question_body_lengths.append(len(lower_text(process_html(X.iloc[i]['question_body']))))
            answer_body_lengths.append(len(lower_text(process_html(X.iloc[i]['body']))))
    draw_lens_histogram(question_body_lengths, 'вопрос', 'question_lengths')
    draw_lens_histogram(answer_body_lengths, 'ответ', 'answer_lengths')
    print(np.mean(question_body_lengths))
    print(np.mean(answer_body_lengths))
    print(np.median(question_body_lengths))
    print(np.median(answer_body_lengths))


def test_features_comparison(csv_path):
    data_reader = ImbalancedDataReader(csv_path, 'question_id')

    correct_ari = []
    correct_fre = []
    correct_si = []
    correct_fkg = []
    correct_cli = []
    correct_gf = []
    correct_lix = []

    incorrect_ari = []
    incorrect_fre = []
    incorrect_si = []
    incorrect_fkg = []
    incorrect_cli = []
    incorrect_gf = []
    incorrect_lix = []

    cnt = 0
    for X, y in data_reader.get_raw_data_labels_batch(data_reader.get_ids(), 50):
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
        for i in range(50):
            if y[i] == 1:
                correct_ari.append(X.iloc[i]['ari'])
                correct_fre.append(X.iloc[i]['fre'])
                correct_si.append(X.iloc[i]['si'])
                correct_fkg.append(X.iloc[i]['fkg'])
                correct_cli.append(X.iloc[i]['cli'])
                correct_gf.append(X.iloc[i]['gf'])
                correct_lix.append(X.iloc[i]['lix'])
            else:
                incorrect_ari.append(X.iloc[i]['ari'])
                incorrect_fre.append(X.iloc[i]['fre'])
                incorrect_si.append(X.iloc[i]['si'])
                incorrect_fkg.append(X.iloc[i]['fkg'])
                incorrect_cli.append(X.iloc[i]['cli'])
                incorrect_gf.append(X.iloc[i]['gf'])
                incorrect_lix.append(X.iloc[i]['lix'])
    draw_histogram(correct_ari, incorrect_ari, 'ARI')
    draw_histogram(correct_fre, incorrect_fre, 'FRE')
    draw_histogram(correct_si,  incorrect_si,  'SI')
    draw_histogram(correct_fkg, incorrect_fkg, 'FKG')
    draw_histogram(correct_cli, incorrect_cli, 'CLI')
    draw_histogram(correct_gf,  incorrect_gf,  'GF')
    draw_histogram(correct_lix, incorrect_lix, 'LIX')


test_features_comparison('SO_data/Posts_sf.csv')
