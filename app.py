import io
import streamlit as st
import pandas as pd
import requests
import nltk
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')  # Downloading NLP models

model = SentenceTransformer('all-mpnet-base-v2') # Loading pre-trained SentenceTransformer model

def load_data():
    url = 'https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv' # given data-set
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text))

    # Select relevant columns and  Combining Description and Genres for semantic search
    data = data[['title', 'description', 'genres']]
    data['combined_text'] = data['description'].fillna('') + ' ' + data['genres'].fillna('')
    data['combined_text'] = data['combined_text'].apply(preprocess_text)

    return data

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    return ' '.join(text)

def generate_embeddings(data):
    sentences = data['combined_text'].tolist()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def find_similar_movies(query_embedding, embeddings, data):
    # Calculate cosine similarity between query embedding and movie embeddings to get top 5 similar movies
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:5] 

    # Retrieving the movie titles 
    similar_movies = data.iloc[top_indices.cpu().numpy()]['title'].tolist()

    return similar_movies

# Load data and generate embeddings
data = load_data()
embeddings = generate_embeddings(data)

# User interface title
st.title('Movie Recommendation System')

user_query = st.text_input('Enter your desired movie characteristics (e.g., heartfelt romantic comedy)')

if user_query:
    query_embedding = model.encode(preprocess_text(user_query), convert_to_tensor=True)

    similar_movies = find_similar_movies(query_embedding, embeddings, data)

    st.subheader('Recommended Movies:')
    for movie in similar_movies:
        st.write(movie)
