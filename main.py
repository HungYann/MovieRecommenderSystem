import streamlit as st
import pandas as pd
import numpy as np

#Importing Libraries
import numpy as np
import pandas as pd


#Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)

@st.cache
def load_data():
    data = pd.io.parsers.read_csv('ratings.dat',
        names=['user_id', 'movie_id', 'rating', 'time'],
        engine='python', delimiter='::',encoding='latin-1')
    data = data[data['movie_id'] < 50]

    return data

@st.cache
def load_movie_data():
    movie_data = pd.io.parsers.read_csv('movies.dat',
                                        names=['movie_id', 'title', 'genre'],
                                        engine='python', delimiter='::', encoding='latin-1')
    movie_data=  movie_data[movie_data['movie_id'] < 50]

    return movie_data



data_load_state = st.text('Loading data...')
data = load_data()
movie_data = load_movie_data()
data_load_state.text("Done! (using st.cache)")



#Creating the rating matrix (rows
# as movies, columns as users)
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

#Normalizing the matrix(subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T



#Computing the Singular Value Decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)




#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    'The recommended movie:'
    i = 0;
    for id in top_indexes + 1:
        "number",i,":",movie_data[movie_data.movie_id == id].title.values[0]
        i=i+1




st.subheader('Number of top recommended movies:')
# Some number in the range 0-10
movie_to_filter = st.slider('from 1 to 5', 0, 1, 5)

'your selected:', movie_to_filter,'top movies'

option = st.sidebar.selectbox(
    'Which number do you like best?',
     movie_data["title"]
)


'Current target movie: ', option

movie_data[movie_data.title == option]



#k-principal components to represent movies, movie_id to find recommendations, top_n print n results
k = 50
# movie id
movie_id = movie_data[movie_data.title == option].movie_id.values[0] # (getting an id from movies.dat)



# movie number
top_n = int(movie_to_filter)
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)

#Printing the top N similar movies
print_similar_movies(movie_data, movie_id, indexes)