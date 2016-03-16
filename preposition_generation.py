import nltk
from nltk.parse.stanford import StanfordParser

def find_next_noun(tag_list,prop_ind):
    choices = tag_list[prop_ind+1:]
    length = len(choices)
    for ind in xrange(length):
        if choices[ind][:2] == 'PR' or choices[ind][:2] == 'NN':
            return ind+prop_ind+1


def find_prev_noun(tag_list,prop_ind):
    length = len(tag_list)
    choices = tag_list[::-1][length-prop_ind:]
    length = len(choices)
    for ind in xrange(length):
        if choices[ind][:2] == 'PR' or choices[ind][:2] == 'NN':
            return length-ind-1


def find_verb(tag_list,prop_ind):
    length = len(tag_list)
    choices = tag_list[::-1][length-prop_ind:]
    length = len(choices)
    for ind in xrange(length):
        if choices[ind][:2] == 'VB':
            return length-ind-1

sentences = ["I talked to Sam.", 
             "Professor came from a weird country.", 
             "Sam went to the restroom at 11:30pm.", 
             "I like pictures of Sam.",
             "She will vote for Sanders."]

#identify "to" and other proposition first.
#find 
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    

    #some processing
    tag_list = [t for (w,t) in tagged]
    try:
        prop_ind = tag_list.index('TO')
    except:
        prop_ind = tag_list.index('IN')
    prop_word = tagged[prop_ind][0]
    #Where, who, how, why, what
    prev_noun_ind = find_prev_noun(tag_list,prop_ind)
    prev_noun = tagged[prev_noun_ind][0]

    next_noun_ind = find_next_noun(tag_list,prop_ind)
    next_noun = tagged[next_noun_ind][0]

    verb_ind = find_verb(tag_list,prop_ind)
    verb = tagged[verb_ind][0]

    print sentence
    # print prev_noun
    # print next_noun
    # print verb

    verb_property = tag_list[verb_ind]
    if verb_property == 'VBD':
        do_word = 'did'
    elif verb_property == 'VBP':
        do_word = 'does'
    else:
        do_word = 'do'

    print ' '.join(['What',do_word,prev_noun,verb,tagged[prop_ind][0], '?'])
    print '\n'

    