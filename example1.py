import streamlit as st
from constants import sambanova_key

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

st.title("Celebrity Search Results")

# Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

input_text = st.text_input("Search celebrity name")

llm = ChatOpenAI(
    api_key=sambanova_key,
    base_url="https://api.sambanova.ai/v1",
    model="Meta-Llama-3.3-70B-Instruct",
    temperature=0.8
)

# Prompt 1
first_prompt = PromptTemplate(
    input_variables=["history", "name"],
    template="""
Previous Conversation:
{history}

Tell me about celebrity {name}
"""
)

# Prompt 2
second_prompt = PromptTemplate(
    input_variables=["history", "person_info"],
    template="""
Previous Conversation:
{history}

From this information extract only the birth year:

{person_info}
"""
)

# Prompt 3
third_prompt = PromptTemplate(
    input_variables=["history", "dob"],
    template="""
Previous Conversation:
{history}

Mention 5 major world events around {dob}
"""
)

# Chains
chain1 = first_prompt | llm
chain2 = second_prompt | llm
chain3 = third_prompt | llm

if input_text:

    person_info = chain1.invoke({
        "history": st.session_state.chat_history,
        "name": input_text
    })

    dob = chain2.invoke({
        "history": st.session_state.chat_history,
        "person_info": person_info.content
    })

    events = chain3.invoke({
        "history": st.session_state.chat_history,
        "dob": dob.content
    })

    # Save to memory
    st.session_state.chat_history += f"""
User: {input_text}

Celebrity Info:
{person_info.content}

Birth Year:
{dob.content}

Events:
{events.content}

----------------------------------------
"""

    # Display Results
    st.subheader("Celebrity Info")
    st.write(person_info.content)

    st.subheader("Birth Year")
    st.write(dob.content)

    st.subheader("Major Events")
    st.write(events.content)

# Sidebar Memory
with st.sidebar:
    st.header("Conversation Memory")

    st.text_area(
        "History",
        st.session_state.chat_history,
        height=400
    )