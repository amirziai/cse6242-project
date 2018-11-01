from similarity_measures import Similarity as s
from scipy.spatial.distance import dice, euclidean
from scipy.stats import pearsonr
import re
import pandas as pd

#d = pd.read_csv('../resources/datasets/bbc-text.csv')
#data = df['text']
d = ["The dog ran over the cat","the movie is about a dog","the book was made into a movie","this is fun","The dog and the cat are over"]
df = pd.DataFrame(d, columns=['text'])

df_train = s.tokenize_questions(df)
dictionary = s.train_dictionary(df_train)

text_csc = s.get_vectors(df_train,dictionary) #sparse
text_normalized = s.vectors(df_train) #notsparse

cosine_sim_matrix = pd.DataFrame(0.,index=range(len(df_train)),columns=range(len(df_train)))
jaccard_coef_matrix = pd.DataFrame(0.,index=range(len(df_train)),columns=range(len(df_train)))
dice_matrix = pd.DataFrame(0.,index=range(len(df_train)),columns=range(len(df_train)))
euclidean_matrix = pd.DataFrame(0.,index=range(len(df_train)),columns=range(len(df_train)))
pearson_coef_matrix = pd.DataFrame(0.,index=range(len(df_train)),columns=range(len(df_train)))

for i in range(0,len(df_train)):
    for j in range(0,len(df_train)):
        cosine_sim_matrix.at[i,j] = s.get_cosine_similarity(text_csc[i],text_csc[j])[0]
        jaccard_coef_matrix.at[i,j] = s.get_jaccard_sim(df_train['text'][i],df_train['text'][j])
        dice_matrix.at[i,j] = dice(text_normalized[i],text_normalized[j])
        euclidean_matrix.at[i,j] = euclidean(text_normalized[i], text_normalized[j])
        pearson_coef_matrix.at[i,j] = pearsonr(text_normalized[i], text_normalized[j])[0]



