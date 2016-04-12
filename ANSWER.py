QUESTION = 'Was Alessandro Volta a professor of chemistry?'

# Parse the text into a list of sentences
import sys
import nltk.data
import math

article = sys.argv[1]
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

mydoclist = ""
with open(article, 'r') as myfile:
   data = myfile.read().replace('\n', '').decode('utf-8')
   mydoclist = tokenizer.tokenize(data)

from collections import Counter

for doc in mydoclist:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1

########################################
# 2
import string #allows for format()

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return document.split().count(term)

vocabulary = build_lexicon(mydoclist)

doc_term_matrix = []

for doc in mydoclist:
    # # print 'The doc is "' + doc + '"'
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    # # print 'The tf vector for Document %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string)
    doc_term_matrix.append(tf_vector)

    # here's a test: why did I wrap mydoclist.index(doc)+1 in parens?  it returns an int...
    # try it!  type(mydoclist.index(doc) + 1)

# # print 'All combined, here is our master document term matrix: '
# # print doc_term_matrix

########################################
# 3. Normalizing vectors to L2 Norm = 1

import math
import numpy as np

def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))

# # print 'A regular old document term matrix: '
# # print np.matrix(doc_term_matrix)
# # print '\nA document term matrix with row-wise L2 norms of 1:'
# # print np.matrix(doc_term_matrix_l2)

# if you want to check this math, perform the following:
# from numpy import linalg as la
# la.norm(doc_term_matrix[0])
# la.norm(doc_term_matrix_l2[0])

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

my_idf_vector = [idf(word, mydoclist) for word in vocabulary]

# # print 'Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']'
# # print 'The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']'


########################################
# 5.
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

my_idf_matrix = build_idf_matrix(my_idf_vector)

## # print my_idf_matrix


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

# # print vocabulary
# # print np.matrix(doc_term_matrix_tfidf_l2) # np.matrix() just to make it easier to look at

########################################
# Our work

print '[Q] "%s"' % QUESTION

# # print 'The doc is "' + QUESTION + '"'
V = [tf(word, QUESTION) for word in vocabulary]
V1 = np.dot(V, my_idf_matrix)
V2 = l2_normalizer(V1)

# # print 'The tf vector for Document is [%s]' % V2

maxVal = 0
maxIdx = -1

result = np.dot(doc_term_matrix_tfidf_l2,V2)

for i in xrange(len(mydoclist)):
    if result[i] > maxVal:
        # print result[i], i
        maxVal = result[i]
        maxIdx = i

ANSWER = mydoclist[maxIdx]

# Identify easy yes/no questions
# Ref: http://www.isi.edu/natural-language/projects/webclopedia/Taxonomy/YES-NO.html
# Assume the question and answer all have a length at least 1.
first_word = QUESTION.split(' ', 1)[0]

def outputAns(q, a):
    # Get all nouns in q
    q_tokens = nltk.word_tokenize(q)
    q_tagged = nltk.pos_tag(q_tokens)
    q_nouns = []
    for t in q_tagged:
        if (t[1] == 'NN'):
            q_nouns.append(t[0])

    # Get all nouns in a
    a_tokens = nltk.word_tokenize(a)
    a_tagged = nltk.pos_tag(a_tokens)
    a_nouns = []
    for t in a_tagged:
        if (t[1] == 'NN'):
            a_nouns.append(t[0])

    # Naive: Go through the nouns and compare if they are
    # the same
    shorter = min(len(q_nouns), len(a_nouns))
    diff = 0
    for i in xrange(shorter):
        aN = a_nouns[i]
        qN = q_nouns[i]
        if (aN != qN):
            print "[A] No"
            return

    print "[A] Yes"

if (first_word == 'Is') or (first_word == 'Does') or (first_word == 'Did') \
   or (first_word == 'Do') or (first_word == 'Was') or (first_word == 'Are')\
   or (first_word == 'Can') or (first_word == 'Has') or (first_word == 'Have')\
   or (first_word == 'Will'):

   outputAns(QUESTION, ANSWER)

else:
    print '[A] "%s"\n' % ANSWER
