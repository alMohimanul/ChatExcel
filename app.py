import streamlit as st
import pandas as pd
from pandasai import  SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import os

st.set_page_config(layout="wide")

class StreamlitResponse(ResponseParser):
  def __init__(self, context) -> None:
     super().__init__(context)
  def format_dataframe(self, result):
     st.dataframe(result["value"])
     return
  def format_plot(self, result):
     st.image(result["value"])
     return
  def format_other(self, result):
     st.write(result["value"])
     return
     


def read_data(uploaded_file):
  if uploaded_file.type == "text/csv":
    df = pd.read_csv(uploaded_file)
  elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
    df = pd.read_excel(uploaded_file)
  else:
    st.error("Please upload only CSV or Excel files.")
    return None
  return df

def main():
    load_dotenv()
    title_style = """
    <style>
        .st-title {
            margin-left: 10%;  
            text-align: center; 
            font-weight: bold;   
        }
    </style>
"""
    st.markdown(title_style, unsafe_allow_html=True)
    st.title("Chat with your excel files")
    st.markdown("""
    <style>
    .centered-div {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh;
    }
    .file-uploader {
        width: 100px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="centered-div">', unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader("", type=["csv", "xlsx"], accept_multiple_files=True, key="fileUploader")
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_files:
       col1,col2 = st.columns([1,2])
       with col1:  
        all_dfs = []
        for uploaded_file in uploaded_files:
            df = read_data(uploaded_file)
            if df is not None:
                all_dfs.append(df)
                if all_dfs:
                  df = pd.concat(all_dfs, ignore_index=True)
                  st.write(df.head(20))
               
        with col2:
            st.info("Chat With Your Files ")
            model = ChatGroq(
               model_name = "mixtral-8x7b-32768",
               api_key=os.environ["GROQ_API_KEY"]
            )
            data = SmartDataframe(
               df,
               config={
                  "llm":model,
                  "response_parser":StreamlitResponse},)
            prompt = st.text_area("Enter your prompt")
            if st.button("Generate"):
               if prompt:
                  with st.spinner("Generating Response..."):
                     st.info(prompt)
                     st.info(data.chat(prompt))

if __name__ == '__main__':
    main() 