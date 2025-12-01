from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource   
def load_model(name="all-mpnet-base-v2"):
    return SentenceTransformer(name)