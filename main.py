import streamlit as st
from constants import sambanova_key
from langchain_openai import ChatOpenAI

# Streamlit UI
st.title("LangChain Demo with SambaNova API")

input_text = st.text_input("Search the topic you want")

# LLM
llm = ChatOpenAI(
    api_key=sambanova_key,
    model="Meta-Llama-3.3-70B-Instruct",
    base_url="https://api.sambanova.ai/v1",
    temperature=0.8
)

# Generate response
if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)