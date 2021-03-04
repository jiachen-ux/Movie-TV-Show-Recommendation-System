import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# read file
df = pd.read_csv("netflix_titles.csv")

def get_title_from_index(df,index):
    return df["title"][index]

def get_index_from_title(df,title): 
    movie_index = df[df["title"]==title].index.values.astype(int)[0]
    return movie_index

def combine_features(key_values,df):
    '''
    This function combines the value that I think is necessary
    for each movie/tv show, given by it's column name in key_values.
    '''
    combined = ''
    for key in key_values:
        df[key] = df[key].fillna("")  # replace nah value to empty
        combined += df[key] + " "
    return combined


def get_similar(df, movie_user_likes):
    '''
    Given by a movie/tv show, find similar movie/tv show
    by calculating cosine similaritiy.
    '''
    key_values = ['title','type','director','cast','listed_in','description']

    # combine useful information for each movie/tv show
    df["combined_info"] = combine_features(key_values, df)

    # create count matrix
    count_matrix = CountVectorizer().fit_transform(df["combined_info"])

    # compute cosine similarity
    cos_similarity = cosine_similarity(count_matrix)
    movie_index = get_index_from_title(df, movie_user_likes)
    similar = list(enumerate(cos_similarity[movie_index]))

    sorted_similar_movie = sorted(similar, key=lambda x:x[1], reverse=True)

    return sorted_similar_movie[:10]

movies = get_similar(df, "Sherlock")

def show_movies(df,movies):
    movie_names = []
    for movie in movies:
        movie_names.append(get_title_from_index(df,movie[0]))
    return movie_names
    
movie_names = show_movies(df,movies)
print(movie_names)

