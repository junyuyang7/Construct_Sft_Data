# 功能：(增删改查)
# 1.按照格式上传sftdata数据 
# 2.查看数据库中的sftdata 
# 3.删除数据库中的sftdata 
# 4.修改数据库中的sftdata 

from Script.db.models import PromptModel
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from Script.create_db import DBconnecter
import requests
import streamlit as st
from st_aggrid import AgGrid, JsCode, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from typing import *
import pandas as pd

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")

SFTDATA_UPLOAD = "http://127.0.0.1:8010/sftdata_upload/"
SFTDATA_UPDATE = "http://127.0.0.1:8010/sftdata_update/"
SFTDATA_LIST = "http://127.0.0.1:8010/sftdata_list/"
SFTDATA_DELETE = "http://127.0.0.1:8010/sftdata_delete/"
SFTDATA_SEARCH = "http://127.0.0.1:8010/sftdata_search_keyword/"

def sftdata_base_page():
    def upload_sftdata(inputs, targets, turn, domain_name, task_name, cls_name, prompt, score, history):
        response = requests.post(SFTDATA_UPLOAD, json={'inputs': inputs, 'targets': targets, 'turn': turn, 'domain_name': domain_name, 'task_name': task_name, 'cls_name': cls_name, 'score': score, 'prompt': prompt, 'history': history, 'data_id': None, 'keyword': None})
        if response.status_code == 200:
            st.success("Prompt upload successfully!")
            return response.json().get('status', False), response.json().get('prompt_data', {})
        else:
            st.error("Failed to upload the prompt.")
            return False, []
        
    def list_all_sftdata():
        response = requests.post(SFTDATA_LIST)
        if response.status_code == 200:
            st.success("Prompt list successfully!")
            return response.json().get('data', {})
        else:
            st.error("Failed to list the prompt.")
            return {}
        
    def delete_sftdata(data_id):
        response = requests.post(SFTDATA_DELETE, json={'inputs': None, 'targets': None, 'turn': None, 'domain_name': None, 'task_name': None, 'score': None, 'prompt': None, 'history': None, 'keyword': None, 'data_id': data_id})
        if response.status_code == 200:
            st.success("Prompt delete successfully!")
            return response.json().get('status', False)
        else:
            st.error("Failed to delete the prompt.")
            return False

    def find_sftdata(keyword):
        response = requests.post(SFTDATA_SEARCH, json={'inputs': None, 'targets': None, 'turn': None, 'domain_name': None, 'task_name': None, 'score': None, 'prompt': None, 'history': None, 'keyword': keyword, 'data_id': None})
        if response.status_code == 200:
            st.success("Prompt find successfully!")
            return response.json().get('status', False), response.json().get('data', {})
        else:
            st.error("Failed to find the prompt.")
            return False, {}

    def update_sftdata(inputs, targets, turn, domain_name, task_name, cls_name, prompt, score, history):
        response = requests.post(SFTDATA_UPDATE, json={'inputs': inputs, 'targets': targets, 'turn': turn, 'domain_name': domain_name, 'task_name': task_name, 'cls_name': cls_name, 'score': score, 'prompt': prompt, 'history': history, 'data_id': None, 'keyword': None})
        if response.status_code == 200:
            st.success("Prompt update successfully!")
            return response.json().get('status', False), response.json().get('prompt_data', {})
        else:
            st.error("Failed to update the prompt.")
            return False, []

    def on_mode_change():
        mode = st.session_state.mode
        text = f"请 {mode} "
        st.toast(text)

    # 显示表格并允许编辑    
    def show_editable_grid(df):
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)  # 允许编辑所有列
        gridOptions = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            editable=True,
            allow_unsafe_jscode=True
        )

        return grid_response['data']

    with st.sidebar:
        # button 
        executor = [
            '下载 sftdata',
            '修改 sftdata',
            '删除 sftdata',
            '查找 sftdata'
        ]
        mode = st.selectbox("请选择操作：",
                                executor,
                                index=0,
                                on_change=on_mode_change,
                                key="mode"
                            )

    if mode == '上传prompt':
        domain_name = st.text_input(
            "prompt所属知识领域",
            key="domain_name",
        )
        task_name = st.text_input(
            "prompt所属任务类别",
            key="task_name",
        )
        cls_name = st.text_input(
            "prompt所属任务类别细分",
            key="cls_name",
        )
        model_type = st.text_input(
            "属于Ask、Answer、Topic还是Judge Model",
            placeholder="请输入[Ask Model, Answer Model, Topic Model, Judge Model]",
            key="model_type",
        )
        prompt = st.text_input(
            "prompt",
            key="prompt",
        )
        args = st.text_input(
            "prompt中含有的参数，请用空格分开",
            key="args",
        )
        if st.button("上传prompt"):
            status, resp = upload_prompt(domain_name, task_name, cls_name, model_type, prompt, args)
            if status:
                st.write("上传成功", resp)
    elif mode == "修改prompt":
        data = list_all_prompt()
        data = pd.DataFrame(data)
        # st.dataframe(data)
        edited_df = show_editable_grid(data)

        # if not edited_df.equals(data):
        #     # update_db_from_dataframe(edited_df)
        #     st.success("Database updated successfully!")

        # st.dataframe(edited_df)
        st.subheader('如果需要修改数据，请填写以下信息。')
        prompt_id = st.text_input(
            "prompt id",
            key="prompt_id",
        )
        domain_name = st.text_input(
            "prompt所属知识领域",
            key="domain_name",
        )
        task_name = st.text_input(
            "prompt所属任务类别",
            key="task_name",
        )
        cls_name = st.text_input(
            "prompt所属任务类别细分",
            key="cls_name",
        )
        model_type = st.text_input(
            "属于Ask、Answer、Topic还是Judge Model",
            placeholder="请输入[Ask Model, Answer Model, Topic Model, Judge Model]",
            key="model_type",
        )
        prompt = st.text_input(
            "prompt具体内容",
            key="prompt",
        )
        args = st.text_input(
            "prompt中含有的参数，请用空格分开",
            key="args",
        )
        if st.button("修改prompt"):
            status, resp = update_prompt(prompt_id, domain_name, task_name, cls_name, model_type, prompt, args)
            if status:
                st.write("修改成功", resp)
    elif mode == "删除prompt":
        data = list_all_prompt()
        data = pd.DataFrame(data)
        # st.dataframe(data)
        edited_df = show_editable_grid(data)
        st.subheader('请输入你要删除的数据的prompt_id')
        prompt_id = st.text_input(
            "prompt id",
            key="prompt_id",
        )
        if st.button("确定删除该prompt"):
            status = delete_prompt(prompt_id)
            if status:
                st.write("删除成功")
    elif mode == "查找prompt":
        data = list_all_prompt()
        data = pd.DataFrame(data)
        # st.dataframe(data)
        edited_df = show_editable_grid(data)
        st.subheader('请输入你要找的关键词')
        keyword = st.text_input(
            "输入你要找的关键词",
            key="keyword",
        )
        if st.button("开始查找"):
            status, resp = find_prompt(keyword)
            if status:
                st.write("查找结果如下", resp)