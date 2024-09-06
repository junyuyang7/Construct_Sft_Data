import streamlit as st

def filter_data_page():

    # 初始化 session state 中的对话历史记录
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # 显示对话历史
    st.write("### 对话记录")
    print(st.session_state['history'])
    for chat in st.session_state['history']:
        st.write(chat)

    # 获取用户输入
    user_input = st.text_input("请输入你的信息:")

    if st.button("发送a"):
        # 模拟生成的响应
        response = f"这是对你的回复: {user_input}"

        # 更新对话历史记录
        st.session_state['history'].append(f"你: {user_input}")
        st.session_state['history'].append(f"助手: {response}")

        # 重新显示对话历史
        st.write("### 对话记录")
        for chat in st.session_state['history']:
            st.write(chat)