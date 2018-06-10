import gensim.downloader as api

from utils.word_utils import lower_text

print('was')
data = api.load('wiki-english-20171001')
print('was')
article_count = 0
sentence_count = 0
word_count = 0
for article in data:
    article_count += 1
    if article_count % 100 == 0:
        print(article_count)
    for text in article['section_texts']:
        sentences = text.split('.')
        for sentence in sentences:
            sentence = lower_text(sentence)
            if len(sentence) >= 5:
                sentence_count += 1
                word_count += len(sentence.split())

print(article_count)
print(sentence_count)
print(word_count)