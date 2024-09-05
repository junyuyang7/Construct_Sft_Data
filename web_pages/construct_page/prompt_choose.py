import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
from fastapi import UploadFile
import os
from typing import *
from Script.config import KEYWORD_FILE, DATA_FILE
import pandas as pd
import re
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, AgGridTheme
from st_aggrid.shared import JsCode
import json

UPLOAD_URL = "http://127.0.0.1:8010/upload/"
PROMPT_LIST = "http://127.0.0.1:8010/prompt_list/"

def test_prompt():
    def upload(files, save_path):
        files_upload = [("files", (file.name, file, file.type)) for file in files]
        response = requests.post(UPLOAD_URL, files=files_upload)
        if response.status_code == 200:
            st.success("Files upload successfully!")
            server_filenames = response.json().get("filenames", [])
            # data = [[{key1: xxx, key2: xxx}, {key1: xxx, key2: xxx}], [{key1: xxx, key2: xxx}, {key1: xxx, key2: xxx}], ....]
            data = response.json().get("json_data", [])
        else:
            st.error("Failed to upload files.")
            server_filenames, data = [], []
        
        # 将文件保存到本地指定路径
        local_filenames = save_files_locally(files, save_path)

        # 返回服务器端和本地的文件名
        return server_filenames, data, local_filenames
        
    def save_files_locally(files, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        saved_filenames = []
        for file in files:
            file_path = os.path.join(save_path, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_filenames.append(file_path)
        return saved_filenames
    
    def list_all_prompt(tabel_name):
        response = requests.post(PROMPT_LIST, json={'tabel_name': tabel_name})
        if response.status_code == 200:
            st.success("Prompt list successfully!")
            return response.json().get('data', {})
        else:
            st.error("Failed to list the prompt.")
            return {}
        
    def replace_prompt_keyword(prompt, keywords_data, replacements):
        # 使用正则表达式提取 {$...} 格式的占位符
        text = prompt['prompt']
        pattern = r'\{\$(\w+)\}'
        matches = re.findall(pattern, text)

        def replace_match(match):
            field_name = match.group(1)
            return str(replacements.get(field_name, match.group(0)))  # 用 replacements 字典中的新字符串替换
        
        for key in matches:
            if key not in keywords_data:
                st.warning('选择的prompt中key和keyword文件中的key对不上')

        new_text = re.sub(pattern, replace_match, text)

        return new_text
    
    # 切换模型
    def on_mode_change():
        mode = st.session_state.model_name
        text = f"已切换到 {mode} 模型。"
        st.toast(text)

    # 将数据填入对应的prompt中的{$keyword}
    def get_final_prompt(selected_prompts, data):
        final_prompt_lst = []
        for i, prompt in enumerate(selected_prompts):
            keywords_data = data[i][0].keys()
            prompt_samples = []
            for sample in data[i]:
                new_prompt = replace_prompt_keyword(prompt, keywords_data, sample)
                prompt_samples.append(new_prompt)
            prompt['prompt'] = prompt_samples
            final_prompt_lst.append(prompt)
        return final_prompt_lst
    
    # 1.先展示prompt供用户选择
    tabel_name = st.text_input(
        "是什么类型的prompt [query_prompt, answer_prompt, evaluate_prompt]",
        key="tabel_name",
    )
    if tabel_name in ['query_prompt', 'answer_prompt', 'evaluate_prompt']:
    # 展示数据库中的prompt
        prompt_data = list_all_prompt(tabel_name)
        # data_df = pd.DataFrame(data)
        options = [
            f"{item['prompt']} (domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}, args: {item['args']})"
            for item in prompt_data
        ]
        selected_options = st.multiselect(
            "请为上述的 keywords 文件选择对应的 prompts: ",
            options,
        )
        selected_prompts = []
        if selected_options:
            selected_prompts = [
                item for item in prompt_data if f"{item['prompt']} (domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}, args: {item['args']})" in selected_options
            ]
            # 显示选择的 prompts 及其详细信息
            for prompt_info in selected_prompts:
                st.write(f"Prompt: {prompt_info['prompt']}")

        # 使用 st.file_uploader 上传多个文件并获取 keyword 
        uploaded_files = st.file_uploader("选择keyword文件", accept_multiple_files=True)
        if uploaded_files:
            server_resp, data, local_resp = upload(uploaded_files, KEYWORD_FILE)
            # # 显示上传成功的文件
            st.write("服务器端文件上传成功：", set(server_resp))
            st.write("本地保存的文件路径：", set(local_resp))
            try:
                assert len(selected_prompts) == len(data)
                # 将数据填入对应的prompt中的{$keyword}
            except Exception as e:
                st.warning(f'选择的prompt数量和keyword数量对不上\n {selected_prompts} | {data}')

            final_prompt_lst = get_final_prompt(selected_prompts, data)
            st.session_state['final_prompt_lst'] = final_prompt_lst
            st.write(final_prompt_lst)
    else:
        st.warning('请先输入是什么类型的prompt')