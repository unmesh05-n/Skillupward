import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 

#Data Collection and Preprocessing

#Loading The Data to a CSV File to Pandas Dataframe
movies_data=pd.read_csv('movies.csv')

#Selecting The Relevant Features For Recommendation
selected_features=['genres','keywords','tagline','cast','director']

#Replacing The Null Values With Null String
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')

#Combining All 5 Selected Features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

#Converting The Text Data To Feature Vectors
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

#Getting The Similarity using Cosine Similarity
similarity=cosine_similarity(feature_vectors)

#Getting The Movie Name From The User
movie_name=input('Enter Your Favourite Movie Name: ')

#Creating A List With All The Movie Names Given In The Dataset
list_of_all_titles=movies_data['title'].tolist()

#Finding The Close Match For The Movie Name Given By The User
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
close_match=find_close_match[0]

#Finding The Index Of The Movie Using Title
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]

#Getting The List Of Similar Movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))

#Sorting The Movies Based On Their Similarity Scores
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

#Print The Name Of Similar Movies Based On The Index Of The Movies
print('Movies Suggested For You: \n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<=10):
        print(i,' ',title_from_index)
        i+=1
