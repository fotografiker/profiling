import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from transformers import pipeline


with st.sidebar:
    st.title("Upload")
    choice =  st.radio("Navigation", ["Upload","Profiling","ML","Transformers"])
    st.info("This application is for ML Pipeline")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choice== "Upload":
    st.title("Upload your dataset")
    file = st.file_uploader("Upload your dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
        

if choice == "Profiling":
    st.title("Automated Exploratory Data")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning Model Using pyCaret")
    chosen_target = st.selectbox("Choose the Target Column", df.columns)
   
    
if choice == "Transformers":
    st.title("GPT Neo TRansformer ")
    generator = pipeline("text-generation",model="EleutherAI/gpt-neo-2.7B")
    prompt = st.text_area("Enter prompt")
    if(st.button("Make magic!")):
        res = generator(prompt, max_length=120, do_sample=True, temperature = 0.9)
        st.text(res[0]["generated_text"])
    