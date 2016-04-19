import sentiment_analysis
import nltk

def find_most_positive_sentence(sentence_list):
    confidence_list = sentiment_analysis.analyze(sentence_list)
    max_value = max(confidence_list)
    max_index = confidence_list.index(max_value)
    return sentence_list[max_index]

def find_main_word(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    main_word = None
    for (word, tag) in tagged:
        if tag == 'PRP' or tag[0] == 'N':
            main_word = word
            break
    main_word = main_word.lower()
    if main_word in ['they','theirs']:
        main_word = 'them'
    elif main_word in ['she', 'hers']:
        main_word = 'her'
    elif main_word in ['i', 'my', 'mine']:
        main_word = 'me'
    elif main_word in ['he', 'his']:
        main_word = 'him'
    return main_word

def generate_senti_question(sentence_list):
    # total_len = 
    mp_sentence = find_most_positive_sentence(sentence_list)
    main_word = find_main_word(mp_sentence)
    q_p1 = "Is the reference to "
    q_p2 = " in the sentence '%s'" % mp_sentence
    q_p3 = " positive or negative or neutral?"
    question =  q_p1 + main_word + q_p2 + q_p3
    return question

def answer(sentence):
    return give_pos_neg(question)


# s_l = ["I am awesome.","You are fucking stupid.", 
#        "I hate Sam.", "This is great."]
# print find_most_positive_sentence(s_l)
# sentence = find_most_positive_sentence(s_l)
# print sentence
# print find_main_word(sentence)
# print generate_senti_question(s_l)