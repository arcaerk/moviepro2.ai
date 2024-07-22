import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.markdown("<h1 style='text-align: center;color:blue;'>Moviepro.ai<",unsafe_allow_html=True)
st.write("## Organize your movie scripts, powered by AI")


df=pd.read_csv('new_df.csv')

def get_similar_plots(movie_title):
    movie_df=df[df['Movie Name'].str.match(movie_title)]
    movie_embedding=df.iloc[0]['Embeddings']
    plot_embeddings=df.iloc[:]['Embeddings'].tolist()
    similarities=cosine_similarity([movie_embedding],plot_embeddings)
    sorted_idx=np.argsort(similarities)[::-1]
    needed_idx=list(sorted_idx)[:6]
    titles=[]
    plots=[]
    for i in needed_idx:
        title=df.iloc[i]["Movie Name"]
        titles.append(title)
        plot=df.iloc[i]["Plots"]
        plots.append(plot)
    return titles,plots

st.write("")
if "page" not in st.session_state:
    st.session_state.page=0

def next_page():
    st.session_state.page+=1
def prev_page():
    st.session_state.page-=1
    
page_number=0
last_page=len(df)

prev,middle,next= st.columns([2,10,2])

if st.session_state.page<last_page:
    next.button(">",on_click=next_page)
else:
    next.write("")

if st.session_state.page>0:
    prev.button("<",on_click=prev_page)
else:
    prev.write("")
    
middle.write(f"Page{1+st.session_state.page} of {last_page}")

row=df.iloc[st.session_state.page]
movie_name=row['Movie Name']
link=row['Wiki Link']
plot=row['Plot']
st.header(movie_name)
st.write(link)
st.write(plot)

if st.button('Get similar Plots'):
    titles,plots=get_similar_plots(movie_name)
    for title, link, plot in zip(titles,plots):
        st.header(title)
        st.write(plot)
        st.markdown("""----------""")