from utils.utils import lower_text, process_html


class CsvToFasttextPreprocessor:

    def process_train(self, fasttext_file_name, data_reader, ids, batch_size=50):
        with open(fasttext_file_name, 'w', encoding='utf-8') as fasttext_file:
            for X, y in data_reader.get_raw_data_labels_batch(set(ids), batch_size):
                for i in range(len(y)):
                    fasttext_file.write('__label__{} {}\n'.format(y[i], lower_text(process_html(X['body'].iloc[i]))))

    def process_test(self, fasttext_file_name, data_reader, ids, batch_size=50):
        with open(fasttext_file_name, 'w', encoding='utf-8') as fasttext_file:
            for X, y in data_reader.get_raw_data_labels_batch(set(ids), batch_size):
                for i in range(len(y)):
                    fasttext_file.write(lower_text(process_html(X['body'].iloc[i])) + '\n')