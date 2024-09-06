import streamlit as st

def test_page():

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

    if st.button("发送"):
        # 模拟生成的响应
        response = f"这是对你的回复: {user_input}"

        # 更新对话历史记录
        st.session_state['history'].append(f"你: {user_input}")
        st.session_state['history'].append(f"助手: {response}")

        # 重新显示对话历史
        st.write("### 对话记录")
        for chat in st.session_state['history']:
            st.write(chat)

    # Categories and corresponding benchmarks
    benchmarks = {
        "Examination": ["MMLU", "CMMLU", "C-Eval", "GaokaoBench"],
        "Knowledge": ["Knowledge-Benchmark-1", "Knowledge-Benchmark-2"],
        "Understanding": ["Understanding-Benchmark-1", "Understanding-Benchmark-2"],
        "Reasoning": ["Reasoning-Benchmark-1", "Reasoning-Benchmark-2"],
        "Code": ["Code-Benchmark-1", "Code-Benchmark-2"],
        "Other": ["Other-Benchmark-1", "Other-Benchmark-2"]
    }

    # Sidebar for benchmark categories
    selected_category = st.selectbox("Select a category", list(benchmarks.keys()))

    # Multi-select for benchmarks in the selected category
    selected_benchmarks = st.multiselect(f"Select benchmarks from {selected_category}", benchmarks[selected_category])

    # Show selected benchmarks
    st.write("Selected benchmarks:", selected_benchmarks)

    # 初始化 session_state 中的值
    if 'text_input' not in st.session_state:
        st.session_state['text_input'] = ''

    if 'multiselect' not in st.session_state:
        st.session_state['multiselect'] = []

    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    # 文本输入框，默认值设置为 session_state 中保存的值
    text_value = st.text_input(
        "请输入文本：", 
        value=st.session_state['text_input'],
        key='text_input_key'
    )

    # 每次用户输入时更新 session_state
    st.session_state['text_input'] = text_value

    # 多选框，默认值设置为 session_state 中保存的值
    options = ['选项1', '选项2', '选项3', '选项4']
    selected_options = st.multiselect(
        "请选择：", 
        options, 
        default=st.session_state['multiselect'],
        key='multiselect_key'
    )

    # 每次用户选择时更新 session_state
    st.session_state['multiselect'] = selected_options

    # 文件上传
    uploaded_files = st.file_uploader(
        "上传文件", accept_multiple_files=True
    )

    # 将文件保存到 session_state
    if uploaded_files:
        # 保存每个文件的内容（防止页面刷新后丢失）
        for uploaded_file in uploaded_files:
            st.session_state['uploaded_files'].append({
                'name': uploaded_file.name,
                'content': uploaded_file.read()
            })

    # 显示已上传的文件
    if st.session_state['uploaded_files']:
        st.write("已上传的文件：")
        for file in st.session_state['uploaded_files']:
            st.write(file['name'])

    # 刷新按钮
    if st.button('刷新页面'):
        st.experimental_rerun()

    # 显示用户的输入和选择
    st.write(f"输入的文本：{st.session_state['text_input']}")
    st.write(f"选择的选项：{st.session_state['multiselect']}")


