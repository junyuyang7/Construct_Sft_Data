import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "ChatAgent_RAG WebUI",
        os.path.join("img", "favicon_icon.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/junyuyang7/ChatAgent_RAG',
            'Report a bug': "https://github.com/junyuyang7/ChatAgent_RAG/issues",
            'About': f"""Welcomm to SFT Data Construct platform！"""
        }
    )

    pages = {
        "对话测试": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "数据批量构造": {
            "icon": "chat-dots",
            "func": construct_page,
        },
        "模板库管理": {
            "icon": "hdd-stack",
            "func": prompt_base_page,
        },
        "对话数据管理": {
            "icon": "hdd-stack",
            "func": sftdata_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "chatagent-rag.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：{0.0}</p>""",
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
        pages[selected_page]["func"](api=api, is_lite=is_lite)