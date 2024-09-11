import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
from fastapi import UploadFile
import os
from typing import *
from Server.config import KEYWORD_FILE, DATA_FILE
import pandas as pd
import re
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, AgGridTheme
from st_aggrid.shared import JsCode
import json
from construct_query import test_query

UPLOAD_URL = "http://127.0.0.1:8010/upload/"
PROMPT_LIST = "http://127.0.0.1:8010/prompt_list/"

# query_params = st.experimental_get_query_params()
# current_page = query_params.get('page', ['prompt_choose'])[0]

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
        
        # 确保 data.key() 中含有num
        try:
            for d in data:
                assert 'num' in d[0].keys()
        except:
            st.warning("你的文件中没有 num 字段")

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
        response = requests.post(PROMPT_LIST, json={'tabel_name': tabel_name, 'prompt_id': None})
        if response.status_code == 200:
            st.success("Prompt list successfully!")
            return response.json().get('data', {})
        else:
            st.error("Failed to list the prompt.")
            return {}
        
    def replace_prompt_keyword(prompt, keywords_data, replacements):
        def replace_match(match):
            field_name = match.group(1)
            return str(replacements.get(field_name, match.group(0)))  # 用 replacements 字典中的新字符串替换
        
        def process_prompts(prompts, pattern, replace_match):
            """处理给定的 prompt 列表，匹配正则并替换"""
            matches = []
            for key in prompts:
                # 查找匹配项并去重
                prompt_matches = re.findall(pattern, prompts[key])
                matches.extend(prompt_matches)
                # 替换匹配项
                prompts[key] = re.sub(pattern, replace_match, prompts[key])
            # 返回去重后的所有匹配项
            return list(set(matches)), prompts
        
        pattern = r'\{\$(\w+)\}'

        # 使用正则表达式提取 {$...} 格式的占位符
        if st.session_state['tabel_name'] == 'all_prompt':
            prompts_dict = {
                'first_query_prompt': prompt['first_query_prompt'],
                'query_prompt': prompt['query_prompt'],
                'answer_prompt': prompt['answer_prompt'],
                'evaluate_prompt': prompt['evaluate_prompt']
            }
            matches, new_prompt = process_prompts(prompts_dict, pattern, replace_match)
        else:
            matches, new_prompt = process_prompts({'prompt': prompt['prompt']}, pattern, replace_match)
            
        for key in matches:
            if key not in keywords_data:
                st.warning('选择的prompt中key和keyword文件中的key对不上')

        return new_prompt

    # 将数据填入对应的prompt中的{$keyword}
    def get_final_prompt(selected_prompts: List[dict], data: List[dict]) -> List[dict]:
        final_prompt_lst = []
        all_num_lst = []
        for i, prompt in enumerate(selected_prompts):
            num_lst = []
            keywords_data = data[i][0].keys()
            if st.session_state['tabel_name'] == 'all_prompt':
                prompt_samples = [[], [], [], []]
                keys = ['first_query_prompt', 'query_prompt', 'answer_prompt', 'evaluate_prompt']
            else:
                prompt_samples = [[]]
                keys = ['prompt']
            for sample in data[i]:
                num_lst.append(int(sample['num']))
                new_prompt = replace_prompt_keyword(prompt, keywords_data, sample)
                for j, key in enumerate(keys):
                    prompt_samples[j].append(new_prompt[key])

            if st.session_state['tabel_name'] == 'all_prompt':
                prompt['first_query_prompt'] = prompt_samples[0]
                prompt['query_prompt'] = prompt_samples[1]
                prompt['answer_prompt'] = prompt_samples[2]
                prompt['evaluate_prompt'] = prompt_samples[3]
            else:
                prompt['prompt'] = prompt_samples[0]
            
            all_num_lst.append(num_lst)
            final_prompt_lst.append(prompt)
        return final_prompt_lst, all_num_lst
    
    # 1.先展示prompt供用户选择
    tabel_name = st.text_input(
        "是什么类型的prompt [query_prompt, answer_prompt, evaluate_prompt, all_prompt]",
        key="tabel_name",
    )
    if tabel_name in ['query_prompt', 'answer_prompt', 'evaluate_prompt', 'all_prompt']:
        # 展示数据库中的prompt
        prompt_data = list_all_prompt(tabel_name, )
        # data_df = pd.DataFrame(prompt_data)
        
        options = [
            f"domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}"
            for item in prompt_data
        ]
        selected_options = st.multiselect(
            "请为上述的 keywords 文件选择对应的 prompts: ",
            options,
        )
        selected_prompts = []
        if selected_options:
            selected_prompts = [
                item for item in prompt_data if f"domain_name: {item['domain_name']}, task_name: {item['task_name']}, cls_name: {item['cls_name']}" in selected_options
            ]
            # 显示选择的 prompts模板 及其详细信息
            for prompt_info in selected_prompts:
                if tabel_name == 'all_prompt':
                    st.write(f"First Query_Prompt: {prompt_info['first_query_prompt']}") 
                    st.write(f"Query_Prompt: {prompt_info['query_prompt']}") 
                    st.write(f"Answer_Prompt: {prompt_info['answer_prompt']}")
                    st.write(f"Evaluate_Prompt: {prompt_info['evaluate_prompt']}")
                else:
                    st.write(f"Prompt: {prompt_info['prompt']}")

        # 使用 st.file_uploader 上传多个文件并获取 keyword 
        uploaded_files = st.file_uploader("选择keyword文件", accept_multiple_files=True)
        is_file = uploaded_files is not None

        if is_file:
            server_resp, data, local_resp = upload(uploaded_files, KEYWORD_FILE)
            # # 显示上传成功的文件
            # st.write("服务器端文件上传成功：", set(server_resp))
            # st.write("本地保存的文件路径：", set(local_resp))
            
            final_prompt_lst, all_num_lst = get_final_prompt(selected_prompts, data)
            st.session_state['final_prompt_lst'] = final_prompt_lst
            st.session_state['all_num_lst'] = all_num_lst # 知晓每个种类需要多少条指令，标语后续生成轮数的数量
            
            final_prompt_df_lst = []
            for final_prompt in final_prompt_lst:
                final_prompt_df = pd.DataFrame({
                    'first_query_prompt': final_prompt['first_query_prompt'],
                    'query_prompt': final_prompt['query_prompt'],
                    'answer_prompt': final_prompt['answer_prompt'],
                    'evaluate_prompt': final_prompt['evaluate_prompt'],
                    })
                final_prompt_df_lst.append(final_prompt_df)
                
            if 'final_prompt_df_lst' not in st.session_state:
                st.session_state['final_prompt_df_lst'] = final_prompt_df_lst
                
            for i, prompt_df in enumerate(final_prompt_df_lst):
                with st.expander(f"这是{final_prompt_lst[i]['domain_name']}的构造数据集"):
                    st.dataframe(prompt_df, width=800, height=300)
        
        # if st.button('开始构造数据'):
        #     test_query()
    else:
        st.warning('请先输入是什么类型的prompt')