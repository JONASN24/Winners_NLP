# import os
import sys
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
################################### HELPER FUNCTIONS ###########################
# From: http://stackoverflow.com/questions/30664677/extract-list-of-
# persons-and-organizations-using-stanford-ner-tagger-in-nltk
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


# Import Stanford NER package
os.environ['JAVAHOME'] = '/Library/Java/JavaVirtualMachines/jdk1.8.0_25.jdk/Contents/Home'
os.environ['CLASSPATH'] = '/Users/jonathanzeng/Desktop/senior_spring/11411/Project/Winners_NLP/lib/stanford-ner-2015-04-20/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/Users/jonathanzeng/Desktop/senior_spring/11411/Project/Winners_NLP/lib/stanford-ner-2015-04-20/classifiers/'

from nltk.tag.stanford import StanfordNERTagger

# Parse the sentence
# sentence = sys.argv[1]
sentence = "Messi is the best"

nertagger = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
neroutput = nertagger.tag(sentence.split())

ne_tree = stanfordNE2tree(neroutput)

ne_in_sent = []
for subtree in ne_tree:
    if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
        ne_label = subtree.label()
        ne_string = " ".join([token for token, pos in subtree.leaves()])
        ne_in_sent.append((ne_string, ne_label))
print ne_in_sent
