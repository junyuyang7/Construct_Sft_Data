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
import random

LOAD_MODEL_URL = "http://127.0.0.1:8010/switch_model/"
CHAT_URL = "http://127.0.0.1:8010/chat/"
CONSTRUCT_SFT_DATA = "http://127.0.0.1:8010/construct_dialog/"

def test_query():
    def load_model(model_name):
        response = requests.post(LOAD_MODEL_URL, json={'model_name': model_name})
        if response.status_code == 200:
            st.success(f"成功切换到 {model_name}")
        else:
            st.error(f"模型切换失败: {response.text}")

    def on_mode_change():
        mode = st.session_state.model_name
        text = f"已切换到 {mode} 模型。"
        st.toast(text)

    def config_aggrid(df: pd.DataFrame) -> GridOptionsBuilder:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(enabled=True)
        gb.configure_side_bar()
        gb.configure_default_column(editable=True, filter=True)

        # 配置允许多行删除的按钮
        gb.configure_selection('multiple', use_checkbox=True)
        return gb
    
    def construct_sft_data(final_prompt_lst, model_name):
        response = requests.post(CONSTRUCT_SFT_DATA, json={'final_prompt_lst': final_prompt_lst, 'model_name': model_name})
        
        if response.status_code == 200:
            return response.json().get('ans_df_lst', [pd.DataFrame({})]), response.json().get('filter_prompt_lst', [{}])
        else:
            return [pd.DataFrame({})], [{}]
        
    try:
        final_prompt_lst = st.session_state['final_prompt_lst']
        final_prompt_df_lst = st.session_state['final_prompt_df_lst']
        all_num_lst = st.session_state['all_num_lst']
        len_prompt_lst = len(final_prompt_df_lst)
        
        assert len(final_prompt_df_lst) == len(final_prompt_lst), "final_prompt_df_lst 和 final_prompt_lst 不一样长"
    except:
        st.warning('你需要先选好Prompt')
        
    # 将所有的prompt数据调用LLM 接口进行批量构造数据并返回xlsx文件
    with st.sidebar:
        # Button to load the model
        global model_name
        model_list = ['chatglm3-6b', 'llama-2-7b-chat', 'llama-2-13b-chat', 'llama-3-8b-instruct']
        model_name = st.selectbox("请选择模型：",
                                model_list,
                                index=0,
                                on_change=on_mode_change,
                                key="model_name"
                            )
        
        if st.button("Load Model"):
            load_model(model_name)

    turn_range = st.slider('选择需要构造的对话轮数', min_value=0, max_value=20, value=(5, 9))
    st.write("<h2 style='text-align: center; font-size: 16px; color: gray;'>以下是待进行数据构建的prompt</h2>", unsafe_allow_html=True)

    for i in range(len_prompt_lst):
        turn_lst = []
        for num_class in all_num_lst:
            # eg: num_class = [2, 2, 4] turn = [[3,3], [4,2], [2,5,3,6]]
            turn = [[random.randint(turn_range[0], turn_range[1]) for _ in range(num)] for num in num_class]
            turn_lst.append(turn)

        final_prompt_df_lst[i]['turn'] = turn_lst
        final_prompt_lst[i]['turn'] = turn_lst
        final_prompt_lst[i]['chat'] = []  # 这个字段用来储存对话数据
        with st.expander(f"这是{final_prompt_lst[i]['domain_name']}的构造数据集"):
            st.dataframe(final_prompt_df_lst[i], height=300, width=800)
    
    
    st.session_state['final_prompt_lst'] = final_prompt_lst

    if st.button("开始构造数据"):
        # 进行批量构造数据并返回xlsx文件(保存至数据库中)，以json的格式
        ans_df_lst, filter_prompt_lst = construct_sft_data(final_prompt_lst, model_name)
        for i, ans_df in enumerate(ans_df_lst):
            with st.expander(f"这是{final_prompt_lst[i]['domain_name']}的构造数据集"):
                st.dataframe(ans_df, width=800, height=300)