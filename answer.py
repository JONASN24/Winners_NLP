#TODO: match root only. E.g. die/died => die

import sys
import nltk.data
import math
from collections import Counter
import string #allows for format()
import numpy as np

YN_words = ['Is','Does','Did','Do','Was','Are','Can','Has','Have','Will']
critical_tags = ['NN', 'CD', 'NNP']

#########################
# process article
#########################
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# open article file
sentences = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        if line != '\n':
            # data = line.replace('\n', '')
            data = line.replace('\n', '').decode('utf-8')
            sentences += tokenizer.tokenize(data)

for s in sentences:
    tf = Counter()
    for word in s.split():
        tf[word] +=1

# 2
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return document.split().count(term)

vocabulary = build_lexicon(sentences)

doc_term_matrix = []

for doc in sentences:
    # # print 'The doc is "' + doc + '"'
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    # # print 'The tf vector for Document %d is [%s]' % ((sentences.index(doc)+1), tf_vector_string)
    doc_term_matrix.append(tf_vector)


# 3. Normalizing vectors to L2 Norm = 1
def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))

########################################
# 4.

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

my_idf_vector = [idf(word, sentences) for word in vocabulary]


########################################
# 5.
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

my_idf_matrix = build_idf_matrix(my_idf_vector)


########################################
# 6.
doc_term_matrix_tfidf = []

#performing tf-idf matrix multiplication
for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

#normalizing
doc_term_matrix_tfidf_l2 = []
for tf_vector in doc_term_matrix_tfidf:
    doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))


########################################
# Identify easy yes/no questions
# Ref: http://www.isi.edu/natural-language/projects/webclopedia/Taxonomy/YES-NO.html
# Assume the question and answer all have a length at least 1.

def answer(question):
    V = [tf(word, question) for word in vocabulary]
    V1 = np.dot(V, my_idf_matrix)
    V2 = l2_normalizer(V1)
    result = np.dot(doc_term_matrix_tfidf_l2,V2)

    # matching string = the sentence that is most similar to the question
    maxVal = 0
    maxIdx = -1
    for i in xrange(len(sentences)):
        if result[i] > maxVal:
            maxVal = result[i]
            maxIdx = i
    matching_string = sentences[maxIdx]

    # check first word to detect yes/no questions
    first_word = question.split(' ', 1)[0]

    ####################
    # (1) Ordinary question
    ####################
    if (first_word not in YN_words):
       return matching_string



    tokens = nltk.word_tokenize(question)
    tagged = nltk.pos_tag(tokens)

    ####################
    # (2) either/or question
    ####################
    if 'or' in tokens:
        i = tokens.index('or')
        if tokens[i-1] in matching_string:
            return tokens[i-1] + '.'
        else:
            return tokens[i+1] + '.'

    ####################
    # (3) Yes/No question
    ####################
    # parse the question to obtain critical information

    critical_info = set([])
    for t in tagged:
        if (t[1] in critical_tags and t[0] not in YN_words):
            critical_info.add(t[0])

    # any critical info in question must appear in matching string.
    for x in critical_info:
        if x not in matching_string:
            return 'No.'
    if 'not' in matching_string:
        return 'No.'
    else:
        return 'Yes.'

####################
# print answers
####################
with open ('questions.txt', 'r') as q, open('output.txt', 'w+') as a:
    for question in q:
        print '[Q] %s' % question.replace("\n", "")
        print '[A] %s' % answer(question)
