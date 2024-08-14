import streamlit as st
import requests

# Define the FastAPI endpoints
LOAD_MODEL_URL = "http://127.0.0.1:8000/load_model/"
CHAT_URL = "http://127.0.0.1:8000/chat/"

# Function to load the model
def load_model():
    response = requests.post(LOAD_MODEL_URL)
    if response.status_code == 200:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load the model.")

# Function to send a message to the chat endpoint
def chat(message):
    response = requests.post(CHAT_URL, json={"text": message})
    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    else:
        return "Failed to get a response."

# Streamlit interface
st.title("LLM Chatbot Interface")

# Button to load the model
if st.button("Load Model"):
    load_model()

# Text input for chat
user_input = st.text_input("You:", "")

# Display chat response
if st.button('send'):
    response = chat(user_input)
    st.text_area("Response:", value=response, height=100)
