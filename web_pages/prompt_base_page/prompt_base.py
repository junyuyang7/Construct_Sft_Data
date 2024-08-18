# 功能：(增删改查)
# 1.按照格式上传prompt模板 
# 2.查看数据库中的模板 
# 3.删除数据库中的模板 
# 4.修改数据库中的模板

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

PROMPT_UPLOAD = "http://127.0.0.1:8010/prompt_upload/"
PROMPT_UPDATE = "http://127.0.0.1:8010/prompt_update/"
PROMPT_LIST = "http://127.0.0.1:8010/prompt_list/"
PROMPT_DELETE = "http://127.0.0.1:8010/prompt_delete/"
PROMPT_SEARCH = "http://127.0.0.1:8010/prompt_search_keyword/"

def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb

def prompt_base_page():
    # 按照格式上传prompt模板 
    def upload_prompt(domain_name, task_name, cls_name, model_type, prompt, args):
        response = requests.post(PROMPT_UPLOAD, json={'domain_name': domain_name, 'task_name': task_name, 'cls_name': cls_name, 'model_type': model_type, 'prompt': prompt, 'args': args, 'prompt_id': None, 'keyword': None})
        if response.status_code == 200:
            st.success("Prompt upload successfully!")
            return response.json().get('status', False), response.json().get('prompt_data', {})
        else:
            st.error("Failed to upload the prompt.")
            return False, []

    def list_all_prompt():
        response = requests.post(PROMPT_LIST)
        if response.status_code == 200:
            st.success("Prompt list successfully!")
            return response.json().get('data', {})
        else:
            st.error("Failed to list the prompt.")
            return {}

    def delete_prompt(prompt_id):
        response = requests.post(PROMPT_DELETE, json={'domain_name': None, 'task_name': None, 'cls_name': None, 'model_type': None, 'prompt': None, 'args': None,'keyword': None, 'prompt_id': prompt_id})
        if response.status_code == 200:
            st.success("Prompt delete successfully!")
            return response.json().get('status', False)
        else:
            st.error("Failed to delete the prompt.")
            return False

    def find_prompt(keyword):
        response = requests.post(PROMPT_SEARCH, json={'domain_name': None, 'task_name': None, 'cls_name': None, 'model_type': None, 'prompt': None, 'args': None, 'keyword': keyword, 'prompt_id': None})
        if response.status_code == 200:
            st.success("Prompt find successfully!")
            return response.json().get('status', False), response.json().get('data', {})
        else:
            st.error("Failed to find the prompt.")
            return False, {}

    def update_prompt(prompt_id, domain_name, task_name, cls_name, model_type, prompt, args):
        response = requests.post(PROMPT_UPDATE, json={'domain_name': domain_name, 'task_name': task_name, 'cls_name': cls_name, 'model_type': model_type, 'prompt': prompt, 'args': args, 'keyword': None, 'prompt_id': prompt_id})
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
            '上传prompt',
            '修改prompt',
            '删除prompt',
            '查找prompt'
        ]
        mode = st.selectbox("请选择操作：",
                                executor,
                                index=0,
                                on_change=on_mode_change,
                                key="mode"
                            )
# domain_name, task_name, cls_name, model_type, prompt
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


