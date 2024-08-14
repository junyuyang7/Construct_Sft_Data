# app.py
import streamlit as st
import requests

# 设置 Streamlit 应用的标题
st.title("交互式对话窗口")

# 创建一个文本输入框
user_input = st.text_input("请输入你的消息:")

if st.button("发送"):
    # 发送消息到 FastAPI 后端
    response = requests.post("http://127.0.0.1:8000/chat/", json={"text": user_input})
    response_data = response.json()
    
    # 显示 FastAPI 后端的响应
    st.write("对话系统回应:", response_data["response"])