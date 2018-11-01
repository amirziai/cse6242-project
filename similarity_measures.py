import numpy as np
import pandas as pd
import re, gensim
from gensim import corpora
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.feature_extraction.text import CountVectorizer

#https://www.kaggle.com/hengqujushi/cosine-similarity-nlp

class Similarity:
    #train dictionary
    def tokenize_questions(df):

        words = re.compile(r"\w+", re.I)
        stopword = stopwords.words('english')

        text_tokenized = []

        for q in df.text.tolist():
            text_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

        df["text_tokenized"] = text_tokenized

        return df

    def train_dictionary(df):
        text_tokenized_list = df.text_tokenized.tolist()
        dictionary = corpora.Dictionary(text_tokenized_list)
        #dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000000)
        dictionary.compactify()

        return dictionary

    ##Create Vectors
    def get_vectors(df, dictionary):
        text_vector = [dictionary.doc2bow(text) for text in df.text_tokenized.tolist()]
        text_csc = gensim.matutils.corpus2csc(text_vector, num_terms=len(dictionary.token2id))

        #print(text_csc)
        return text_csc.transpose()

    ##get cosine similarity
    def get_cosine_similarity(t1, t2):
        cosine_sim = []
        for i,j in zip(t1,t2):
            sim = cs(i,j)
            cosine_sim.append(sim[0][0])

        return cosine_sim

    def vectors(df):
        text = [t for t in df['text']]
        vectorizer = CountVectorizer(text) #look at tdifVectorizer
        vectorizer.fit(text)
        vectorizer.transform(text).toarray()
        return [t / np.sqrt(sum(i * i for i in t)) for t in vectorizer.transform(text).toarray()]

    #jaccard coefficient
    def get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / len(a.union(b))