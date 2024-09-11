# 功能：(增删改查)
# 1.按照格式上传prompt模板 
# 2.查看数据库中的模板 
# 3.删除数据库中的模板 
# 4.修改数据库中的模板

# from Server.db.models import PromptModel
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
# from Server.create_db import DBconnecter
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
    def upload_prompt(tabel_name, domain_name, task_name, cls_name, args=None, first_query_args=None, query_args=None, answer_args=None, evaluate_args=None, first_query_prompt=None, query_prompt=None, answer_prompt=None, evaluate_prompt=None, prompt=None):
        response = requests.post(PROMPT_UPLOAD, json={
            'tabel_name': tabel_name, 
            'domain_name': domain_name, 
            'task_name': task_name, 
            'cls_name': cls_name, 
            'prompt': prompt, 
            'args': args, 
            'first_query_prompt': first_query_prompt, 
            'query_prompt': query_prompt, 
            'first_query_args': first_query_args, 
            'query_args': query_args, 
            'answer_prompt': answer_prompt, 
            'answer_args': answer_args, 
            'evaluate_prompt': evaluate_prompt, 
            'evaluate_args': evaluate_args, 
            'prompt_id': None, 
            'keyword': None})
        if response.status_code == 200:
            st.success("Prompt upload successfully!")
            return response.json().get('status', False), response.json().get('prompt_data', {})
        else:
            st.error("Failed to upload the prompt.")
            return False, []

    def list_all_prompt(tabel_name):
        response = requests.post(PROMPT_LIST, json={'tabel_name': tabel_name, 'prompt_id': None})
        if response.status_code == 200:
            st.success("Prompt list successfully!")
            return response.json().get('data', {})
        else:
            st.error("Failed to list the prompt.")
            return {}

    def delete_prompt(tabel_name, prompt_id):
        response = requests.post(PROMPT_DELETE, json={'tabel_name': tabel_name, 'prompt_id': prompt_id})
        if response.status_code == 200:
            st.success("Prompt delete successfully!")
            return response.json().get('status', False)
        else:
            st.error("Failed to delete the prompt.")
            return False

    # TODO
    def find_prompt(tabel_name, keyword):
        response = requests.post(PROMPT_SEARCH, json={'tabel_name': tabel_name, 'domain_name': None, 'task_name': None, 'cls_name': None, 'prompt': None, 'args': None, 'keyword': keyword, 'prompt_id': None})
        if response.status_code == 200:
            st.success("Prompt find successfully!")
            return response.json().get('status', False), response.json().get('data', {})
        else:
            st.error("Failed to find the prompt.")
            return False, {}

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
    
    def check_prompt_args(args_prompt, args):
        try:
            args = args.split(';')[0]
            assert args.split() == args_prompt
            return True
        except Exception as e:
            st.warning(f'prompt中的args -- {args_prompt}和你输入的args -- {args}不对应')
            return False

    import re
    def get_prompt_keyword_input(text):
        # 使用正则表达式提取 {$...} 格式的占位符，可以直接进行填充的参数
        pattern = r'\{\$(\w+)\}'
        matches = re.findall(pattern, text)
        return matches
    
    def get_prompt_keyword_iter(text):
        # 使用正则表达式提取 {$$...} 格式的占位符，在构造的过程中才能进行填充
        pattern = r'\{\$\$(\w+)\}'
        matches = re.findall(pattern, text)
        return matches

    with st.sidebar:
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
    if mode == '上传prompt' or mode == '修改prompt':
        tabel_name = st.text_input(
            "是什么类型的prompt [query_prompt, answer_prompt, evaluate_prompt, all_prompt]",
            key="tabel_name",
        )
        if tabel_name in ['query_prompt', 'answer_prompt', 'evaluate_prompt', 'all_prompt']:
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
            if mode == '修改prompt':
                data = list_all_prompt(tabel_name, )
                data = pd.DataFrame(data)
                st.dataframe(data, width=800, height=400)
                st.subheader('如果需要修改数据，请填写以下信息。')
            if tabel_name == 'all_prompt':
                first_query_prompt = st.text_input("first_query_prompt(必须包含{$num})", key="first_query_prompt",)
                
                query_prompt = st.text_input("query_prompt", key="query_prompt",)
                
                answer_prompt = st.text_input("answer_prompt", key="answer_prompt",)
                
                evaluate_prompt = st.text_input("evaluate_prompt", key="evaluate_prompt",)
                
                first_query_args = st.text_input("first_query_prompt 中含有的参数，请用空格分开同种类型的(必定要有num，表示你需要生成的first_query数量)，用;(英文)分开不同类型的", key="first_query_args",)
                
                query_args = st.text_input("query_prompt 中含有的参数，请用空格分开同种类型的，用;(英文)分开不同类型的", key="query_args",)
                
                answer_args = st.text_input("answer_prompt 中含有的参数，请用空格分开同种类型的，用;(英文)分开不同类型的", key="answer_args",)
                
                evaluate_args = st.text_input("evaluate_prompt 中含有的参数，请用空格分开同种类型的，用;(英文)分开不同类型的", key="evaluate_args",)
            else:
                prompt = st.text_input("prompt", key="prompt",)
                
                args = st.text_input("prompt中含有的参数，请用空格分开同种类型的，用;(英文)分开不同类型的", key="args",)

            if st.button("上传prompt"):
                # 确保args和prompt中的对应 
                if tabel_name == 'all_prompt':
                    query_args_input = get_prompt_keyword_input(query_prompt)
                    answer_args_input = get_prompt_keyword_input(answer_prompt)
                    evaluate_args_input = get_prompt_keyword_input(evaluate_prompt)
                    if_same1, if_same2, if_same3 = check_prompt_args(query_args_input, query_args), check_prompt_args(answer_args_input, answer_args), check_prompt_args(evaluate_args_input, evaluate_args)
                    if_same = if_same1 and if_same2 and if_same3
                else:
                    args_input = get_prompt_keyword_input(prompt)
                    if_same = check_prompt_args(args_input, args)
                if if_same:
                    if tabel_name == 'all_prompt':
                        status, resp = upload_prompt(
                            tabel_name=tabel_name, 
                            domain_name=domain_name, 
                            task_name=task_name, 
                            cls_name=cls_name,  
                            query_args=query_args, 
                            first_query_args=first_query_args, 
                            answer_args=answer_args, 
                            evaluate_args=evaluate_args, 
                            query_prompt=query_prompt, 
                            first_query_prompt=first_query_prompt, 
                            answer_prompt=answer_prompt, 
                            evaluate_prompt=evaluate_prompt,)
                    else:
                        status, resp = upload_prompt(
                            tabel_name=tabel_name, 
                            domain_name=domain_name, 
                            task_name=task_name, 
                            cls_name=cls_name, 
                            prompt=prompt, 
                            args=args)
                    if status:
                        st.write("上传成功", resp)
                else:
                    st.warning('你args这一栏写错了')
        else:
            st.warning('请先输入是什么类型的prompt')

    elif mode == "删除prompt":
        tabel_name = st.text_input(
            "是什么类型的prompt [query_prompt, answer_prompt, evaluate_prompt, all_prompt]",
            key="tabel_name",
        )
        if tabel_name in ['query_prompt', 'answer_prompt', 'evaluate_prompt', 'all_prompt']:
            data = list_all_prompt(tabel_name, )
            data = pd.DataFrame(data)
            st.dataframe(data, width=800, height=400)
            st.subheader('请输入你要删除的数据的prompt_id')
            prompt_id = st.text_input(
                "prompt id",
                key="prompt_id",
            )
            if st.button("确定删除该prompt"):
                status = delete_prompt(tabel_name, prompt_id)
                if status:
                    st.write("删除成功")
        else:
            st.warning('请先输入是什么类型的prompt')
    elif mode == "查找prompt":
        tabel_name = st.text_input(
            "是什么类型的prompt [query_prompt, answer_prompt, evaluate_prompt, all_prompt]",
            key="tabel_name",
        )
        if tabel_name in ['query_prompt', 'answer_prompt', 'evaluate_prompt', 'all_prompt']:
            data = list_all_prompt(tabel_name, )
            data = pd.DataFrame(data)
            st.dataframe(data, width=800, height=400)
            edited_df = show_editable_grid(data)
            st.subheader('请输入你要找的关键词')
            keyword = st.text_input(
                "输入你要找的关键词",
                key="keyword",
            )
            if st.button("开始查找"):
                status, resp = find_prompt(tabel_name, keyword)
                if status:
                    st.write("查找结果如下", resp)
        else:
            st.warning('请先输入是什么类型的prompt')


