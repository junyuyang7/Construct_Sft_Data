import streamlit as st
import requests
from streamlit_chatbox import ChatBox
from streamlit_option_menu import option_menu
from web_pages.dialogue_page.dialogue import dialogue_page
from web_pages.construct_page.construct import construct_page
from web_pages.prompt_base_page.prompt_base import prompt_base_page
from web_pages.dialogue_page.dialogue import sftdata_base_page
import os
import sys
import uuid

# def construct_page():
#     st.title("Construct Data page")
    
if __name__ == "__main__":
    st.set_page_config(
        "Construct SFT Data",
        os.path.join("img", "favicon.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://google.com',
            'Report a bug': "https://google.com",
            'About': f"""欢迎使用 Construct SFT Data WebUI！"""
        }
    )

    pages = {
        "对话测试": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "批量构造": {
            "icon": "hdd-stack",
            "func": construct_page,
        },
        "Prompt管理": {
            "icon": "hdd-stack",
            "func": prompt_base_page,
        },
        "SFT数据管理": {
            "icon": "hdd-stack",
            "func": sftdata_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "favicon.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：0</p >""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        # pages[selected_page]["func"](api=api)
        pages[selected_page]["func"]()