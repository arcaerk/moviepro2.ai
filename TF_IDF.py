import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd

st.markdown("<h1 style='text-align:centre;color:blue;'>Moviepro.ai</h1>",unsafe_allow_html=True)
st.write('## Extract Keywords and Simiilar movies with TF-IDF')
movie_plots_path="American_Movie_Plots_2005_2021_v1.csv"
@st.cache_data

def load_data():
    df=pd.read_csv(movie_plots_path)
    df.drop_duplicates(subset='Wiki Link', keep=False, inplace=True)
    return df

@st.cache_data

def get_transformed_output():
    df=load_data()
    tfdidfvectorizer=TfidfVectorizer(stop_words='english',analyzer='word')
    tfdidfvectorizer.fit(df['Plot'])
    transform_output=tfdidfvectorizer.transform(df['Plot'])
    return tfdidfvectorizer,transform_output

def get_keywords(movie_title:str):
    tfidfvectorizer,transform_output=get_transformed_output()
    df=load_data()
    movie_df=df[df['Movie Name'].str.match(movie_title)]
    movie_plot=movie_df['Plot']
    movie_vector=tfidfvectorizer.transform(movie_plot)
    ftr_names=tfidfvectorizer.get_feature_names_out()
    feature_array=np.array(ftr_names)
    sorted_indices = np.argsort(movie_vector.toarray()[0])[::-1]  # Sort in descending order
    top_10_keywords = feature_array[sorted_indices]
    return list(top_10_keywords)[:10]
def get_similar_movies(movie_title:str):
    tfidfvectorizer,transform_output=get_transformed_output()
    df=load_data()
    movie_df=df[df['Movie Name'].str.match(movie_title)]
    index = movie_df.iloc[0][0]
    search_vec=transform_output[index]
    cosine_similarities = linear_kernel(search_vec, transform_output).flatten()
    related_docs_indices=np.argsort(cosine_similarities)[::-1]
    related_docs_indices = list(related_docs_indices)[:10]
    related_docs_indices.remove(index)
    titles=[]
    plots=[]
    links=[]
    for ind in related_docs_indices:
        title=df.iloc[ind]['Movie Name']
        plot=df.iloc[ind]['Plot']
        link=df.iloc[ind]['Wiki Link']
        titles.append(title)
        plots.append(plot)
        links.append(link)
    return titles,links,plots

data=load_data()

st.write("")
if "page" not in st.session_state:
    st.session_state.page=0

def next_page():
    st.session_state.page+=1
def prev_page():
    st.session_state.page-=1
    
page_number=0
last_page=4000
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

row=data.iloc[st.session_state.page]
movie_name=row['Movie Name']
link=row['Wiki Link']
plot=row['Plot']
st.header(movie_name)
st.write(link)
st.write(plot)

if st.button('Get Keywords'):
    keywords=get_keywords(movie_name)
    keywords=keywords[:10]
    listTostr=' || '.join(str(elem) for elem in keywords)
    st.write(listTostr)

if st.button('Get similar Plots'):
    titles,links,plots=get_similar_movies(movie_name)
    for title, link, plot in zip(titles,links,plots):
        st.header(title)
        st.write(link)
        st.write(plot)
        st.markdown("""----------""")