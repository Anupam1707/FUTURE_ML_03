import streamlit as st
import joblib
import json
import random
import socket
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Hybrid Chatbot", layout="centered")
st.title("💬 AI-Powered Support Chatbot")
st.markdown("""
<style>
div.stChatMessage.user {
    background-color: #e6f2ff;
}
div.stChatMessage.bot {
    background-color: #f0f0f0;
}
.message-bubble {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

model = joblib.load("chatbot_intent_model.pkl")

with open("Intents.json") as f:
    data = json.load(f)
intent_responses = {
    intent["tag"]: intent.get("responses", [])
    for intent in data["intents"]
}

def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False

try:
    llm = ChatGoogleGenerativeAI(
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        model="gemini-2.0-flash",
        temperature=0.3
    )
except:
    llm = None

CONFIDENCE_THRESHOLD = 0.6


def get_chatbot_response(user_input):
    proba = model.predict_proba([user_input])[0]
    max_index = proba.argmax()
    confidence = proba[max_index]
    predicted_tag = model.classes_[max_index]

    fallback_response = "I'm sorry, I didn't understand that."
    base_response = random.choice(intent_responses.get(predicted_tag, [fallback_response]))

    if is_connected() and llm:
        try:
            prompt = f"""
        You are a support assistant.
        
        The user's query is:
        \"\"\"{user_input}\"\"\"
        
        The intent classifier suggests this response:
        \"\"\"{base_response}\"\"\"
        
        Do not mention the classifier or repeat the input. Do not acknowledge the query. 
        Return ONLY the improved support response in a clear, friendly tone.
        If needed Blindly answer as well
        """
            messages = [HumanMessage(content=prompt)]
            reply = llm.invoke(messages).content.strip()
            return reply
        except:
            return base_response
    else:
        return base_response

with st.container():
    for entry in st.session_state.chat_history:
        st.markdown(f"<div class='message-bubble stChatMessage user'><b>You:</b> {entry['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message-bubble stChatMessage bot'><b>Bot:</b> {entry['bot']}</div>", unsafe_allow_html=True)

with st.form("chat_form"):
    user_input = st.text_input("You:", placeholder="Type your message and press Enter")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    reply = get_chatbot_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": reply})
    st.rerun()

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()