# 功能：
# 1.上传关键字对应的文件，并决定是否需要保存 
# 2.解析文件中关键字的数据，并将数据填入对应的prompt中的{$keyword} 
# 3.text框填写prompt（或者再prompt_base_page中选择对应的prompt） 
# 4.获取所有的prompt数据进行批量构造数据并返回xlsx文件(prompt, answer) 
# 5.对返回的xlsx文件进行在线编辑，删除、修改等

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
CHAT_URL = "http://127.0.0.1:8010/chat/"
UPLOAD_DIRECTORY = "test"

def construct_page():
    st.title("Construct Data page")
    st.text('上传文件格式需要是 jsonl 格式')

    def load_model():
        response = requests.post(CHAT_URL)
        if response.status_code == 200:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load the model.")

    def chat(text, model_name, message=[]):
        response = requests.post(CHAT_URL, json={"text": text, "model_name": model_name, "message": message})
        if response.status_code == 200:
            return response.json().get("response", "No response received.")
        else:
            return "Failed to get a response."

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
    
    def list_all_prompt():
        response = requests.post(PROMPT_LIST)
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
    
    def config_aggrid(df: pd.DataFrame) -> GridOptionsBuilder:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(enabled=True)
        gb.configure_side_bar()
        gb.configure_default_column(editable=True, filter=True)

        # 配置允许多行删除的按钮
        gb.configure_selection('multiple', use_checkbox=True)
        return gb

    def construct_answer(final_prompt_lst):
        ans_df_lst = []
        for i, prompt in enumerate(final_prompt_lst):
            prom_lst, ans_lst, hist_lst = [], [], []
            for sample in prompt['prompt']:
                ans = chat(sample, model_name)
                # hist = [{'role': 'user', 'content': sample}, {'role': 'assistant', 'content': ans}]
                hist = json.dumps([{'role': 'user', 'content': sample}, {'role': 'assistant', 'content': ans}], ensure_ascii=False)
                prom_lst.append(sample)
                ans_lst.append(ans)
                hist_lst.append(hist)
            ans_df = pd.DataFrame({'prompt': prom_lst, 'answer': ans_lst, 'history': hist_lst})
            ans_df_lst.append(ans_df)

        return ans_df_lst

    # 使用 st.file_uploader 上传多个文件并获取 keyword 
    uploaded_files = st.file_uploader("选择文件", accept_multiple_files=True)
    if uploaded_files:
        server_resp, data, local_resp = upload(uploaded_files, KEYWORD_FILE)
        # 显示上传成功的文件
        st.write("服务器端文件上传成功：", set(server_resp))
        st.write("本地保存的文件路径：", set(local_resp))
    
        # 获取prompt：1.自己在text框书写，2.从数据库中选择
        # prompt_method = st.radio("选择获取 prompt 的方式：", ("自己写", "从数据库中选择"))

        # 展示数据库中的prompt
        prompt_data = list_all_prompt()
        # data_df = pd.DataFrame(data)
        options = [
            f"{item['prompt']} (domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}, model_type: {item['model_type']}, args: {item['args']})"
            for item in prompt_data
        ]

        # 选择 prompt 
        if len(options) > 0:
            # 使用 st.multiselect 让用户选择多个 prompt，并保持选择顺序
            selected_options = st.multiselect(
                "请为上述的 keywords 文件选择对应的 prompts: ",
                options,
            )
            selected_prompts = []
            if selected_options:
                selected_prompts = [
                    item for item in prompt_data if f"{item['prompt']} (domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}, model_type: {item['model_type']}, args: {item['args']})" in selected_options
                ]
                # 显示选择的 prompts 及其详细信息
                for prompt_info in selected_prompts:
                    st.write(f"Prompt: {prompt_info['prompt']}")
        
            # 需要保证选择的prompt数量和上传的keywords文件数量一致
            if st.button('选好了'):
                try:
                   assert len(selected_prompts) == len(data)
                    # 将数据填入对应的prompt中的{$keyword}
                except Exception as e:
                    st.warning(f'选择的prompt数量和keyword数量对不上\n {selected_prompts} | {data}')

            final_prompt_lst = get_final_prompt(selected_prompts, data) 

            # 将所有的prompt数据调用LLM 接口进行批量构造数据并返回xlsx文件
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

            # 进行批量构造数据并返回xlsx文件
            ans_df_lst = construct_answer(final_prompt_lst)
            for i, ans_df in enumerate(ans_df_lst):
                prompt = final_prompt_lst[i]
                # 设置 AgGrid 的配置
                gb = config_aggrid(ans_df)
                grid_options = gb.build()
                if st.button('开始进行构造'):
                    # 对返回的xlsx文件进行在线编辑，删除、修改等
                    with st.expander(f"编辑数据集 {i + 1} - {prompt['domain_name']}"):
                        if 'edited_df' not in st.session_state:
                            st.session_state['edited_df'] = ans_df.copy()
                        response = AgGrid(
                            ans_df,
                            gridOptions=grid_options,
                            update_mode=GridUpdateMode.VALUE_CHANGED,
                            allow_unsafe_jscode=True,
                            theme=AgGridTheme.ALPINE  # 可选主题
                        )
                        # 获取用户编辑后的数据
                        edited_df = pd.DataFrame(response['data'])
                        if 'selected_rows' not in st.session_state:
                            st.session_state['selected_rows'] = response['selected_rows']
                        st.session_state['edited_df'] = edited_df

                # 显示用户删除的行
                if st.button("删除选中的行"):
                    if 'selected_rows' in st.session_state:
                        selected_rows = st.session_state['selected_rows']
                        st.write(selected_rows)  # 显示选中的行
                        
                        if selected_rows:
                            selected_indices = [int(row['_selectedRowNodeInfo']['nodeRowIndex']) for row in selected_rows]
                            edited_df = st.session_state['edited_df'].drop(st.session_state['edited_df'].index[selected_indices])
                            st.session_state['edited_df'] = edited_df

                            # 重新渲染表格
                            with st.expander(f"编辑数据集"):
                                response = AgGrid(
                                    edited_df,
                                    gridOptions=st.session_state['grid_options'],
                                    update_mode=GridUpdateMode.VALUE_CHANGED,
                                    allow_unsafe_jscode=True,
                                    theme=AgGridTheme.ALPINE  # 可选主题
                                )
                                st.write("行已删除并重新渲染表格")
                    else:
                        st.write("没有选中的行")

                # 将编辑后的数据保存为 Excel 文件
                if st.button(f'保存数据集 {i + 1} 为 Excel'):
                    save_path = f"{DATA_FILE}/{prompt['domain_name']}-{prompt['task_name']}-{prompt['cls_name']}-{prompt['model_type']}.xlsx"
                    os.makedirs(DATA_FILE, exist_ok=True)
                    edited_df = st.session_state['edited_df']
                    edited_df.to_excel(save_path, index=False)
                    print('1222222')
                    st.success(f"数据已保存到 {save_path}")
                    st.download_button(
                        label="下载编辑后的 Excel 文件",
                        data=edited_df.to_excel(index=False, engine='openpyxl'),
                        file_name=f"edited_data_{i + 1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        else:
            st.warning("Prompt 库中没有可用的 prompt。")
        

