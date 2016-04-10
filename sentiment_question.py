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