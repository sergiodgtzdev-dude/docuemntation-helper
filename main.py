from dotenv import load_dotenv

load_dotenv()
import hashlib
from typing import Optional, Set

import streamlit as st

st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
from io import BytesIO

import requests
from PIL import Image


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_profile_picture(email: str) -> Optional[Image.Image]:
    """Fetch a small avatar with strict timeouts so UI never blocks."""
    email_norm = (email or "").strip().lower().encode("utf-8")
    email_md5 = hashlib.md5(email_norm).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_md5}?d=identicon&s=200"
    try:
        response = requests.get(
            gravatar_url,
            timeout=(2.0, 4.0),  # (connect, read)
            headers={"User-Agent": "documentation-helper/1.0"},
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None


# Custom CSS for dark theme and modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #252526;
    }
    .stMessage {
        background-color: #2D2D2D;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Set page config at the very beginning


# Sidebar user information
with st.sidebar:
    st.title("User Profile")

    # You can replace these with actual user data
    user_name = "John Doe"
    user_email = "john.doe@example.com"

    profile_pic = get_profile_picture(user_email)
    if profile_pic is not None:
        st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("LangChain🦜🔗 Udemy Course- Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Use a form to prevent reruns until submission
with st.form(key="chat_form"):
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")
    submit_clicked = st.form_submit_button("Submit")

if submit_clicked and prompt:
    with st.spinner("Generating response..."):
        # Lazy import so app can render instantly even if backend init is slow.
        from backend.core import run_llm

        generated_response = run_llm(query=prompt)

        sources = set(doc.metadata["source"] for doc in generated_response["context"])
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)


# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
