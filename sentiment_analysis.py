import nltk
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import pickle

def word_feats(words):
    accepted_tags = {'VB','RB',"AD"}
    tags = nltk.pos_tag(words)
    tag_dict = dict()
    for (word,tg) in tags:
        tag_dict[word] = tg
    result_dict = dict()
    for word in words:
        cur_tg = tag_dict[word]
        if len(cur_tg) >= 2 and cur_tg[:2] in accepted_tags:
            result_dict[word] = True
    return result_dict

def bigram_feats(words):
    result_dict = dict()
    length = len(words)
    for i in xrange(length-1):
        feature = words[i] + ' ' + words[i+1]
        result_dict[feature] = True
    return result_dict

def unigram_train():
    negfeats = []
    posfeats = []
    with open("data/training.txt", 'r') as train_file:
        for line in train_file:
            split_res = line.split('\t', 1)
            score = split_res[0]
            sentence = split_res[1]
            words = re.sub("[^\w]", " ",  sentence).split()
            #neg
            if score == '0':
                negfeats.append((word_feats(words),0)) 
            #pos
            else:
                posfeats.append((word_feats(words),1)) 
    trainfeats = negfeats + posfeats
    classifier = NaiveBayesClassifier.train(trainfeats)
    return classifier
   
def bigram_train():
    negfeats = []
    posfeats = []
    with open("data/rt-polarity.neg", 'r') as train_file:
        for line in train_file:
            words = re.sub("[^\w]", " ",  line).split()
            #neg
            negfeats.append((bigram_feats(words),0))
    with open("data/rt-polarity.pos", 'r') as train_file:
        for line in train_file:
            words = re.sub("[^\w]", " ",  line).split()
            #pos
            posfeats.append((bigram_feats(words),1))
    trainfeats = negfeats + posfeats
    classifier = NaiveBayesClassifier.train(trainfeats)
    return classifier

# unigram_classifier = unigram_train()
# bigram_classifier = bigram_train()

# f = open('unigram_classifier.pickle', 'wb')
# pickle.dump(unigram_classifier, f)
# f.close()

# f = open('bigram_classifier.pickle', 'wb')
# pickle.dump(bigram_classifier, f)
# f.close()

def analyze_sentence(sentence, unigram_classifier, bigram_classifier):
    words = re.sub("[^\w]", " ",  sentence).split()
    unigram_feature = word_feats(words)
    bigram_feature = bigram_feats(words) 
    unigram_dist = unigram_classifier.prob_classify(unigram_feature)
    bigram_dist = bigram_classifier.prob_classify(bigram_feature)
    uni_positive_confidence = unigram_dist.prob(1)
    bi_positive_confidence = bigram_dist.prob(1)

    if uni_positive_confidence > 0.7 or uni_positive_confidence < 0.3:
        return uni_positive_confidence
    else:
        linear_confidence = uni_positive_confidence * 0.3 \
                        + bi_positive_confidence * 0.7
        return linear_confidence

#return a single positive confidence if input is a sentence
#return a list of pos confidence if input is a list of sentence
def analyze(raw_text):
    f = open('unigram_classifier.pickle', 'rb')
    unigram_classifier = pickle.load(f)
    f.close()
    f = open('bigram_classifier.pickle', 'rb')
    bigram_classifier = pickle.load(f)
    f.close()
    if type(raw_text) is str:
        pos = analyze_sentence(raw_text,unigram_classifier,bigram_classifier)
        return pos
    elif type(raw_text) is list:
        result = []
        for sentence in raw_text:
            cur_pos = analyze_sentence(sentence,unigram_classifier,bigram_classifier)
            result.append(cur_pos)
        return result

def give_pos_neg(sentence):
    confidence = analyze(sentence)
    if confidence > 0.7:
        return "Positive."
    elif (confidence >= 0.3 and confidence < 0.7):
        return "Neutral."
    else:
        return "Negative."


# print analyze(["I am awesome.","You are fucking stupid.", 
#        "I hate Sam.", "Wrong."])
# print analyze("I like you so much.")
# print analyze("I hate you.")
# print analyze("I hate you so much.")
# print analyze("Messi failed at that season.")
