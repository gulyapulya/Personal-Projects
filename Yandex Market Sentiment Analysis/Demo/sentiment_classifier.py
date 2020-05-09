from sklearn.externals import joblib

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("lr6.pkl")
        self.vectorizer = joblib.load("count.pkl")
        self.classes_dict = {'neg': "negative", 'pos': "positive", -1: "prediction error"}

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)
        except:
            return [-1]

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        return "Review type: {}".format(self.classes_dict[class_prediction]) 