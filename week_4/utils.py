import pickle
import pandas as pd
import json


def predict_model(docs_new):
    f = pd.read_csv("topics.csv",header=None)
    v=dict(zip(f[0].to_list(), f[1].to_list()))
    
    # load the dataset from the official sklearn datasets
    model=pickle.load(open('newsclassifier_model.pkl','rb'))
    #v=dict(zip(list(y), pandasDF['topic'].to_list()))
    
    # Prediction
    #docs_new=['The World Health Organization (WHO) announced 26 proposed members to an advisory committee aimed to steer studies into the origin of the COVID-19 pandemic and other pathogens of epidemic potential.']

    predicted=model.predict([docs_new])
    print(v[predicted[0]])
    return v[predicted[0]]

predict_model("The World Health Organization (WHO) announced 26 proposed members to an advisory committee aimed to steer studies into the origin of the COVID-19 pandemic and other pathogens of epidemic potential")

    