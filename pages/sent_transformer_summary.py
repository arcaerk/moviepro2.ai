import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

st.markdown("<h1 style='text-align: center;color:blue;'>Moviepro.ai</h1>", unsafe_allow_html=True)
st.write("## Organize your movie scripts, powered by AI")
@st.cache_data
def load_data():
    df=pd.read_csv('new_df.csv')
    return df

df=load_data()

# Convert embeddings from string representation back to numpy arrays
def parse_embeddings(embedding_str):
    return np.array(ast.literal_eval(embedding_str))

# Apply parsing function to the Embeddings column
df['Embeddings'] = df['Embeddings'].apply(parse_embeddings)

def get_similar_plots(movie_title):
    movie_df = df[df['Movie Name'].str.match(movie_title)]
    
    if movie_df.empty:
        return [], []
    
    movie_embedding = movie_df.iloc[0]['Embeddings']
    plot_embeddings = df['Embeddings'].tolist()
    similarities = cosine_similarity([movie_embedding], plot_embeddings)
    
    # Get top 6 most similar plots
    sorted_idx = np.argsort(similarities.flatten())[::-1][:6]
    
    titles = []
    plots = []
    for i in sorted_idx:
        titles.append(df.iloc[i]["Movie Name"])
        plots.append(df.iloc[i]["Plot"])
        
    return titles, plots

st.write("")
if "page" not in st.session_state:
    st.session_state.page = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

page_number = 0
last_page = len(df) - 1

prev, middle, next = st.columns([2, 10, 2])

if st.session_state.page < last_page:
    next.button(">", on_click=next_page)
else:
    next.write("")

if st.session_state.page > 0:
    prev.button("<", on_click=prev_page)
else:
    prev.write("")

middle.write(f"Page {1 + st.session_state.page} of {last_page + 1}")

row = df.iloc[st.session_state.page]
movie_name = row['Movie Name']
link = row['Wiki Link']
plot = row['Plot']
st.header(movie_name)
st.write(link)
st.write(plot)

if st.button('Get similar Plots'):
    titles, plots = get_similar_plots(movie_name)
    if titles:
        for title, plot in zip(titles, plots):
            st.header(title)
            st.write(plot)
            st.markdown("""----------""")
    else:
        st.write("No similar plots found.")
