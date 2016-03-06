# NOTE: This script takes a single sentence and generate a yes/no question based
# on the sentence
import nltk
import sys
import en

# Parse the sentence
sentence = sys.argv[1]

tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

# For debugging
print(tagged)

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
                    print('Is ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?')
                # 2. Does
                else:
                    print('Does ' + ' '.join(tokens[:i]) + ' ' + en.verb.present(te[0]) + ' ' +
                        ' '.join(tokens[i+1:len(tokens)-1]) + '?')
            # Past tense
            elif t == 'VBD':
                # Was
                if (te[0] == 'was'):
                    print('Was ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?')
                else:
                    print('Did ' + ' '.join(tokens[:i]) + ' ' + en.verb.present(te[0]) + ' ' +
                        ' '.join(tokens[i+1:len(tokens)-1]) + '?')
            # Modal
            elif t == 'MD':
                # Capitalize first letter
                print(tokens[i].title() + ' ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens)-1]) + '?')

            # Present tense
            elif t == 'VBP':
                # Are
                if (te[0] == 'are'):
                    print('Are ' + ' '.join(tokens[:i]) + ' ' + ' '.join(tokens[i+1:len(tokens) - 1]) + '?')
                else:
                    print('Do ' + ' '.join(tokens[:len(tokens) - 1]) + '?')
