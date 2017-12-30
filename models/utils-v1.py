import os
import json
import pandas as pd
import dill as pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_and_train():
    df = pd.read_csv("../data/filtered_labeled_train_40.csv")

    X_train = df['description'].values.astype('U')
    y_train = df['job_class'].values
    job_names = df.job_title.astype('category').unique()

    pipe = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1), stop_words='english')),
        ('clf', MultinomialNB(alpha=4e-03))
        ])

    clf = pipe.fit(X_train, y_train)

    return clf


if __name__ == '__main__':
    model = build_and_train()

    filename = 'model-v1.pk'
    with open("../models/"+filename, 'wb') as file:
        pickle.dump(model, file, protocol=2)
