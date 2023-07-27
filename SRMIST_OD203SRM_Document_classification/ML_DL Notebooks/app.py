# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:42:08 2023

@author: soumil
"""

import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
import sklearn
import numpy as np
import os
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim
from gensim.models import Word2Vec
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os


def levDistance(string1, string2):
    m, n = len(string1), len(string2)
    curr = list(range(n+1))
    for i in range(m):
        prev, curr = curr, [i+1] + [0] * n
        for j in range(n):
            curr[j+1] = prev[j] if string1[i] == string2[j] else min(curr[j], prev[j], prev[j+1]) + 1
    return curr[n]


def cosDistance(string1, string2):
    vector1 = sum([model[word] for word in string1.split() if word in model]) / len(string1.split())
    vector2 = sum([model[word] for word in string2.split() if word in model]) / len(string2.split())
    cosine_sim = 1 - spatial.distance.cosine(vector1, vector2)
    return cosine_sim


SVM = pickle.load(open('SVM', 'rb'))
SGD = pickle.load(open('SGD', 'rb'))
BernoulliNB = pickle.load(open('BernoulliNB', 'rb'))
LogReg = pickle.load(open('LogReg', 'rb'))
vectorizer = pickle.load(open('vectorizer','rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer','rb'))
categories = ['insurance', 'entertainment', 'finance',' ', 'travel', 'medical']
template_directory = "templates"
files = os.listdir(template_directory)
template_dict = {'entertainment':set(), 'finance':set(), 'travel':set(), 'insurance':set(), 'medical':set()}

def dist_predict(test):
    max_dist = 0
    max_dist_label = ''
    for x in template_dict:
        for y in template_dict[x]:
            dist = cosDistance(test, y)
            if(dist > max_dist):
                max_dist = dist
                max_dist_label = x
    return max_dist_label

def predictlev(test):
    min_dist = 100000000
    min_dist_label = ''
    for x in template_dict:
        for y in template_dict[x]:
            dist = levDistance(test, y)
            if(dist < min_dist):     # minimize levenshtein distance
                min_dist = dist
                min_dist_label = x
    return min_dist_label


def main():
    st.title('Text Classification App')
    input_text = st.text_input('Enter the text of the document here')
    result = {

    }

    if st.button('Run Models'):
        X_new = vectorizer.transform([input_text])
        X_new = tfidf_transformer.transform(X_new)
        y_new = SVM.predict(X_new)

        for x in y_new:
            result['SVM'] = categories[x]
        X_new = vectorizer.transform([input_text])
        X_new = tfidf_transformer.transform(X_new)
        y_new = SGD.predict(X_new)

        for x in y_new:
            result['SGD'] = categories[x]
        X_new = vectorizer.transform([input_text])
        X_new = tfidf_transformer.transform(X_new)
        y_new = BernoulliNB.predict(X_new)

        for x in y_new:
            result['BernoulliNB'] = categories[x]
        X_new = vectorizer.transform([input_text])
        X_new = tfidf_transformer.transform(X_new)
        y_new = LogReg.predict(X_new)


        for x in y_new:
            result['LogReg'] = categories[x]
        result[input_text] = dist_predict(input_text)
        st.success(result)
    
    
    
if __name__ == '__main__':
    for folder_name in files:
        folder_path = os.path.join(template_directory, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                template_content = file.read()
                template_dict[folder_name].add(template_content)
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    main()
    
    
    
    