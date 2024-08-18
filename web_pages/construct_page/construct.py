import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
from fastapi import UploadFile
import os
from typing import *

UPLOAD_URL = "http://127.0.0.1:8010/upload/"
UPLOAD_DIRECTORY = "test"

def construct_page():
    st.title("Construct Data page")

    def upload(files):
        files_upload = [("files", (file.name, file, file.type)) for file in files]
        response = requests.post(UPLOAD_URL, files=files_upload)
        if response.status_code == 200:
            st.success("Files upload successfully!")
            return response.json().get("filenames", [])
        else:
            st.error("Failed to upload files.")
            return []
    
    # 使用 st.file_uploader 上传多个文件
    uploaded_files = st.file_uploader("选择文件", accept_multiple_files=True)
    if uploaded_files:
        resp = upload(uploaded_files)
        st.write("文件上传成功：", set(resp))
    
    # # 检查是否有文件上传
    # if uploaded_files:
    #     for uploaded_file in uploaded_files:
    #         # 读取文件内容
    #         file_details = {
    #             "filename": uploaded_file.name,
    #             "filetype": uploaded_file.type,
    #             "filesize": uploaded_file.size
    #         }

    #         # 在页面上显示文件详情
    #         st.write(file_details)

    #         # 显示文件内容（根据文件类型处理）
    #         if uploaded_file.type == "text/plain":
    #             # 读取并显示文本文件内容
    #             text = uploaded_file.read().decode("utf-8")
    #             st.text_area("文件内容", text)
                
    #         elif uploaded_file.type.startswith("image/"):
    #             # 显示图片文件
    #             st.image(uploaded_file, caption=uploaded_file.name)
    #         else:
    #             st.write("无法显示此文件类型的内容")

