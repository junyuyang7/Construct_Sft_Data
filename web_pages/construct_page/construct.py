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
import sys

sys.path.append(os.path.dirname(__file__))

from construct_query import test_query
from filter_data import filter_data_page
from data_evaluate import test_evaluate
from prompt_choose import test_prompt

UPLOAD_URL = "http://127.0.0.1:8010/upload/"
PROMPT_LIST = "http://127.0.0.1:8010/prompt_list/"
CHAT_URL = "http://127.0.0.1:8010/chat/"
UPLOAD_DIRECTORY = "test"

def construct_page():
    def on_mode_change():
        mode = st.session_state.mode
        text = f"请进行 {mode} "
        st.toast(text)

    st.title("Construct Data page")
    # st.text('选择你要进行的步骤')
    if 'tabel_name' not in st.session_state:
        st.session_state['tabel_name'] = 'query_prompt'

    pages = {
        "prompt选择": {
            "icon": "chat",
            "func": test_prompt,
        },
        "SFT数据构造": {
            "icon": "hdd-stack",
            "func": test_query,
        },
        # "SFT数据筛选": {
        #     "icon": "hdd-stack",
        #     "func": filter_data_page,
        # },
        # "数据评估": {
        #     "icon": "hdd-stack",
        #     "func": test_evaluate,
        # },
    }
    with st.sidebar:
        executor = [
            'prompt选择',
            'SFT数据构造',
            # 'SFT数据筛选',
            # '数据评估',
        ]
        mode = st.selectbox("请选择构造的步骤：", executor, index=0, on_change=on_mode_change, key="mode")

        # options = list(pages)
        # icons = [x["icon"] for x in pages.values()]

        # default_index = 0
        # selected_page = option_menu(
        #     "",
        #     options=options,
        #     icons=icons,
        #     # menu_icon="chat-quote",
        #     default_index=default_index,
        # )

    if mode in pages:
        # pages[selected_page]["func"](api=api)
        pages[mode]["func"]()
        

