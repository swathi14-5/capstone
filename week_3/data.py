import pyspark
import json
import string
import pandas as pd
import swifter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import nltk
from nltk.corpus import stopwords
# from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import LabelEncoder

import re
import warnings



warnings.filterwarnings("ignore")

directory="./jars/*"
modle_path="./newsclassifier_model.pkl"

#nltk.download('stopwords')
#nltk.download('punkt')
#stopwords=stopwords.words('english')

def train_model():
    global v
    spark=SparkSession \
    .builder \
    .appName("NewsClassifier_Datapreparation") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/week2.news_articles") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/week2.news_articles") \
    .config('spark.driver.extraClassPath', directory) \
    .getOrCreate()

    print(spark)

    df=spark.read.format("mongo").load()

    # model prediction
    dataset=df.select(col('topic'), col('title'), col('summary'))
    # dataset.show()
    pandasDF=dataset.toPandas()
    pandasDF["title"].fillna("no_title", inplace=True)
    pandasDF["summary"].fillna("No_summary", inplace=True)
    pandasDF["topic"].fillna("No_topic", inplace=True)
    pandasDF['text']=pandasDF['title'] + pandasDF['summary'] + pandasDF['topic']
    pandasDF['Text_parsed']=pandasDF['text'].apply(process_text)

    # Label Encoding


    X=pandasDF[['Text_parsed']]
    y=pandasDF['topic']

    encoder=LabelEncoder()
    y=encoder.fit_transform(y)

    x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
    v=dict(zip(list((y)), pandasDF['topic'].to_list()))
    with open('topics.csv', 'w') as f:
        for key in v.keys():
            f.write("%d,%s\n"%(key,v[key]))
    print(v)
    #print(list(y))
    """ with open('topic.txt', 'w') as convert_file:
     convert_file.write(json.dumps(v)) """

    # Training the Model
    text_clf=Pipeline(
        [('vect', CountVectorizer(analyzer="word", stop_words="english")), ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultinomialNB(alpha=.01))])

    text_clf.fit(x_train['Text_parsed'].to_list(), list(y_train))

    # Testing Model
    X_TEST=x_test['Text_parsed'].to_list()
    Y_TEST=list(y_test)

    predicted=text_clf.predict(X_TEST)

    np.mean(predicted == Y_TEST)

    # Prediction
    docs_new=['The World Health Organization (WHO) announced 26 proposed members to an advisory committee aimed to steer studies into the origin of the COVID-19 pandemic and other pathogens of epidemic potential.']
    predicted=text_clf.predict(docs_new)
    print(v[predicted[0]])
    print(predicted[0])

#saving the model;
""" with open(modle_path,'wb') as f:
    pickle.dump(text_clf,f)  """


def process_text(text):
    text=text.replace('\n', ' ').replace('\r', '').strip()
    text=re.sub(' +', ' ', text)
    text=re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words=set(stopwords.words('english'))
    word_tokens=word_tokenize(text)
    filtered_sentence=[w for w in word_tokens if not w in stop_words]
    filtered_sentence=[]
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text=" ".join(filtered_sentence)
    return text


train_model()