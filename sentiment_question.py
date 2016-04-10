import sentiment_analysis

def generate(text):
    classifier = sentiment_analysis.train()
    # classifier.show_most_informative_features()
    feature_list = classifier.most_informative_features(20)
    features = [feature for (feature, boolean) in feature_list]
    print features

generate("hello")


def answer(question):
     pass



# negcutoff = len(negfeats)*3/4
# poscutoff = len(posfeats)*3/4
# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
# print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
# print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
