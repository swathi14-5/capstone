import numpy as np
import pickle
#import data
from utils import predict_model
from flask import Flask,render_template,jsonify,request

app=Flask(__name__)
#model=pickle.load(open('newsclassifier_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("new.html")


@app.route("/prediction",methods=["POST","GET"])
def predict():
    news_summary=request.form["summary_text"]
    print(news_summary)
    #param = request.get_json()
    #print(param['text'])
    result=predict_model(news_summary)
    return render_template("output_display_page.html",output=result,Text_recieved=news_summary)


if __name__ =='__main__':
    app.run()