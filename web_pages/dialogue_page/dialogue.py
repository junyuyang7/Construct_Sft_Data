import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
import os
from typing import *
import sys
import uuid

# Define the FastAPI endpoints
LOAD_MODEL_URL = "http://127.0.0.1:8010/switch_model/"
CHAT_URL = "http://127.0.0.1:8010/chat/"

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "favicon.png"
    )
)

def dialogue_page():
    def load_model(model_name):
        response = requests.post(LOAD_MODEL_URL, json={'model_name': model_name})
        if response.status_code == 200:
            st.success(f"成功切换到 {model_name}")
        else:
            st.error(f"模型切换失败: {response.text}")
            
    def on_mode_change():
        mode = st.session_state.model_name
        text = f"请点击 Load Model 切换到 {mode} 模型。"
        st.toast(text)
        
    def chat(text, message):
        response = requests.post(CHAT_URL, json={"text": text, "message": message})
        if response.status_code == 200:
            return response.json().get("response", "No response received."), response.json().get("chat_lst", [{}])
        else:
            return "Failed to get a response."
        

    # Streamlit interface
    # st.title("LLM Chatbot Interface")
    
    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用 [`Construct Data`]! \n\n"
            f"当前运行的模型`chatglm3-6b`, 您可以开始提问了."
        )
        chat_box.init_session()

    # 选择什么模型？
    with st.sidebar:
        # Button to load the model
        global model_name
        
        chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "
        model_list = ['chatglm3-6b', 'llama-2-7b-chat', 'llama-2-13b-chat', 'llama-3-8b-instruct']
        model_name = st.selectbox("请选择模型：",
                                model_list,
                                index=0,
                                on_change=on_mode_change,
                                key="model_name"
                            )
        if 'pre_model_name' not in st.session_state:
            st.session_state['pre_model_name'] = 'test'
        
        if st.button("Load Model"):
            if model_name != st.session_state['pre_model_name']:
                st.session_state['pre_model_name'] = model_name
                load_model(model_name)
            else:
                st.write('模型没有切换')
        
    chat_box.output_messages()
    # 渲染对话框
    if 'chat_lst' not in st.session_state:
        st.session_state['chat_lst'] = []

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        chat_box.user_say(prompt)
        chat_box.ai_say("正在思考...")
        message = st.session_state['chat_lst']
        text, chat_lst = chat(prompt, message)
        st.session_state['chat_lst'] = chat_lst
        print(chat_lst)
        
        chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标
    
    # Text input for chat
    # user_input = st.text_input("You:", "")


    # # Display chat response
    # if st.button('send'):
    #     response = chat(user_input, model_name)
    #     st.text_area("Response:", value=response, height=100)