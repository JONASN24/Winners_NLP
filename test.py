import os
import nltk
import sys
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
# nltk.download('all',halt_on_error=False)
from nltk.parse.stanford import *
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.stem.snowball import SnowballStemmer
from nltk.util import breadth_first
import en
import re

ParserPathPrefix = "C:\Users\y\Documents\courses\Senior Spring\\nlp\stanford-parser-full-2015-04-20"
NERPathPrefix = "C:\Users\y\Documents\courses\Senior Spring\\nlp\stanford-ner-2015-04-20"
POSPathPrefix = "C:\Users\y\Documents\courses\Senior Spring\\nlp\stanford-postagger-2015-04-20"
# slf4PathPrefix = "C:\Users\y\Documents\courses\Senior Spring\\nlp\slf4j-1.7.18"
os.environ['JAVAHOME'] = 'C:\Program Files\Java\jdk1.8.0_73\\bin'  #or your java path
os.environ['CLASSPATH'] = ParserPathPrefix + '\stanford-parser.jar;' + POSPathPrefix + '\stanford-postagger.jar;' + NERPathPrefix + '\stanford-ner.jar;'+ ParserPathPrefix + '\stanford-parser-3.5.2-models.jar;'+ POSPathPrefix + '\models\\'

os.environ['STANFORD_MODELS'] = ParserPathPrefix + '\stanford-parser-3.5.2-models.jar;'+ NERPathPrefix + '\classifiers\\;' + POSPathPrefix + '\models\\'

sentence = sys.argv[1]
# parser=StanfordParser()
# print "the parser output"
# print list(parser.raw_parse(sentence))



#stemmer = SnowballStemmer("english")



tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
questions = ""
# # for part in ne_tree:
# # 	if isinstance(part, nltk.tree.Tree):   
# # 		# label = part.label()
# # 		for i in part:
# # 			t = i[1]
# # 	else:
# # 		t = part[1]


# subject word
subject = ''
if (len(tagged) > 0):
    subject = tagged[0]
    if (subject[1] != 'PRP'):
        # Go through the tags
        for i in xrange(1, len(tagged)):
            te = tagged[i]
            t = te[1]
            if t == 'VBZ':
                # 1. IS
                if (te[0] == 'is'):
                    questions = 'is ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?'
                # 2. Does
                else:
                    questions = 'does ' + ' '.join(tokens[:i]) + ' ' + en.verb.present(te[0]) + ' ' +' '.join(tokens[i+1:len(tokens)-1]) + '?'
            # Past tense
            elif t == 'VBD':
                # Was
                if (te[0] == 'was'):
                    questions = 'was ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?'
                else:
                    questions = 'did ' + ' '.join(tokens[:i]) + ' ' + en.verb.present(te[0]) + ' ' +' '.join(tokens[i+1:len(tokens)-1]) + '?'
            # Modal
            elif t == 'MD':
                # Capitalize first letter
                questions = tokens[i].title() + ' ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?'

            # Present tense
            elif t == 'VBP':
                # Are
                if (te[0] == 'are'):
                    questions = 'are ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens) - 1]) + '?'
                else:
                    questions = 'Do ' + ' '.join(tokens[:len(tokens) - 1]) + '?'

# print questions

nertagger = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
neroutput = nertagger.tag(sentence.split())

# postagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# posoutput = postagger.tag(sentence)
# postoutput = nltk.pos_tag(nltk.word_tokenize(sentence))
# print "POS Tagging output"
# print posoutput


def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree

ne_tree = stanfordNE2tree(neroutput)


ne_in_sent = []
for subtree in ne_tree:
    if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
        ne_label = subtree.label()
        ne_string = " ".join([token for token, pos in subtree.leaves()])
        ne_in_sent.append((ne_string, ne_label))

wh_questions = []
for part in ne_in_sent:
	label  = part[1]
	wh = ''
	if label == 'PERSON':
		wh = 'Who '
	elif label == 'LOCATION':
		wh = 'Where '
	elif label == 'ORGANIZATION':
		wh = 'Which organization '
	elif label == 'DATE' or label == 'TIME':
		wh = 'When '
	if sentence.startswith(part[0]):
		newQ = re.sub(part[0]+' ', '', sentence)
	else:
		newQ = re.sub(part[0]+' ','', questions) 
	newQ = wh + newQ
	wh_questions.append(newQ)
print wh_questions