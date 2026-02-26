import streamlit as st
import pandas as pd
import seaborn as sns
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler   # ← Updated import
st.set_page_config(page_title="Titanic Chat Agent", page_icon="🚢", layout="wide")
st.title("Titanic Dataset Chat Agent")
st.markdown(" Ask anything in plain English!")
st.info("""
**Important note**  
This chatbot is **specialized in the Titanic dataset only**.  
**I cannot answer unrelated questions** (news, jokes, general knowledge, etc.).  
 

""", icon="ℹ️",)

@st.cache_data
def load_titanic():
    return sns.load_dataset("titanic")

df = load_titanic()
st.success(f" Loaded {len(df)} passengers")

# Sidebar for API key
# with st.sidebar:
#     gemini_api_key = st.text_input("Google Gemini API Key (Free)", type="password", help="Get free key from aistudio.google.com")
#     if not gemini_api_key:
#         st.info("Please enter your free Gemini API key to start chatting.")
#         st.stop()


custom_prefix = """
You are a friendly, helpful Titanic data analyst.
When the user asks for any visualization (histogram, bar chart, pie chart, box plot, etc.):
1. Create the plot with matplotlib or seaborn.
2. Save it exactly as 'plot.png' using plt.savefig('plot.png', bbox_inches='tight')
3. Close the figure with plt.close()
4. In your FINAL answer write: "Here is the visualization:" 
Do NOT show code in the final answer unless asked.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Titanic chatbot powered by Gemini Free. Ask me anything! Example: 'Show me age histogram' or 'What % survived by class?'"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask anything about Titanic passengers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    if os.path.exists("plot.png"):
        os.remove("plot.png")

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",     
            temperature=0,
            google_api_key="AIzaSyARR6Ra-dbk2BZinJa6I32FKQ0wpdRMtow",
            streaming=True
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
           prefix=custom_prefix,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

        response = agent.run(prompt, callbacks=[st_callback])

        st.write(response)

        if os.path.exists("plot.png"):
            st.image("plot.png", caption="📊 Generated Visualization", use_column_width=True)
            os.remove("plot.png")   


        st.session_state.messages.append({"role": "assistant", "content": response})


