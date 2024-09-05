import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
import os
from typing import *
import sys
import uuid

# Define the FastAPI endpoints
LOAD_MODEL_URL = "http://127.0.0.1:8010/load_model/"
CHAT_URL = "http://127.0.0.1:8010/chat/"

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "favicon.png"
    )
)

def dialogue_page():
    # Function to load the model
    def load_model():
        response = requests.post(CHAT_URL)
        if response.status_code == 200:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load the model.")
            
    def on_mode_change():
        mode = st.session_state.model_name
        text = f"已切换到 {mode} 模型。"
        st.toast(text)
        
    # Function to send a message to the chat endpoint
    def chat(text, model_name, message):
        response = requests.post(CHAT_URL, json={"text": text, "model_name": model_name, "message": message})
        if response.status_code == 200:
            return response.json().get("response", "No response received.")
        else:
            return "Failed to get a response."
        

    # Streamlit interface
    # st.title("LLM Chatbot Interface")
    
    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用 [`Construct Data`]! \n\n"
            f"当前运行的模型`vivo-BlueLM-TB-Pro-TEST`, 您可以开始提问了."
        )
        chat_box.init_session()

    # 选择什么模型？
    with st.sidebar:
        # Button to load the model
        global model_name
        
        chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "
        model_list = ['vivo-BlueLM-HB-PRE', 'vivo-BlueLM-TB-Pro-TEST', 'Doubao-pro-32k']
        model_name = st.selectbox("请选择模型：",
                                model_list,
                                index=0,
                                on_change=on_mode_change,
                                key="model_name"
                            )
        
        if st.button("Load Model"):
            load_model()
        
    chat_box.output_messages()
    # 渲染对话框
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        chat_box.user_say(prompt)
        chat_box.ai_say("正在思考...")
        message = st.session_state['history']
        text = chat(prompt, model_name, message)
        st.session_state['history'].append({'role': 'user', 'content': prompt})
        st.session_state['history'].append({'role': 'assistant', 'content': text})

        # print(st.session_state['history'])
        # for t in r:
        #     if error_msg := check_error_msg(t):  # check whether error occured
        #         st.error(error_msg)
        #         break
        #     text += t.get("text", "")
        #     chat_box.update_msg(text)
        #     message_id = t.get("message_id", "")

        # metadata = {
        #     "message_id": message_id,
        # }
        chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标
    
    # Text input for chat
    # user_input = st.text_input("You:", "")


    # # Display chat response
    # if st.button('send'):
    #     response = chat(user_input, model_name)
    #     st.text_area("Response:", value=response, height=100)