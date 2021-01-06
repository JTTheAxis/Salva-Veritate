import sys
sys.path.insert(0, r"C:\Users\JXYTi\AppData\Local\Programs\Python\Python38\site-packages")
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("News Dataset.csv")

df.shape
df.head()

labels=df.label
labels.head()

x_train, x_test, y_train, y_test=train_test_split(df["text"], labels, test_size=.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)
print("Accuracy: " + str(round(score*100, 2)) + "%")