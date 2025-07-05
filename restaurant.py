import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

prompt_template = PromptTemplate(
    input_variables=["cuisine"],
    template="""
You are a creative restaurant consultant.

Given the cuisine: {cuisine}

Please provide:

1. A unique and creative restaurant name suitable for that cuisine (avoid generic names).
2. A list of 5 menu items with brief but appetizing descriptions.

Format your answer as:

Restaurant Name: <name>
Menu Recommendations:
- <item 1>: <description>
- <item 2>: <description>
...
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Restaurant Menu Recommender 🍴", page_icon="🌮")
st.title("🌮🍛 AI Restaurant Menu Recommender")
st.caption("Powered by Gemini 2.0 Flash via LangChain ✨")

st.write(
    """
    👉 Choose a cuisine to get:
    - A unique restaurant name idea
    - 5 creative menu recommendations with mouth-watering descriptions
    """
)

cuisines = ["Indian", "Mexican", "Italian", "Chinese", "Thai", "Japanese", "French", "Greek"]
selected_cuisine = st.selectbox("Choose a cuisine:", cuisines)

if st.button("✨ Get Recommendations"):
    with st.spinner("Cooking up ideas with Gemini... 🍳✨"):
        try:
            response = chain.run(cuisine=selected_cuisine)
            st.success("Here are your recommendations! 🌟")
            # st.text_area("🍽️ Result", response, height=300)
            st.markdown(response)
            # clean_response = response.replace("**", "")
            # st.text_area("🍽️ Result", clean_response, height=300)
        except Exception as e:
            st.error(f"❌ Error: {e}")
