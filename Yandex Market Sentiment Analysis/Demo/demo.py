from flask import Flask
from sentiment_classifier import SentimentClassifier
from flask import render_template, request

app = Flask(__name__)
classifier = SentimentClassifier()

@app.route("/model", methods=["POST", "GET"])
def index_page(text="",prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        prediction_message = classifier.get_prediction_message(text)
    return render_template('hello.html', text=text, prediction_message=prediction_message)  
   
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)