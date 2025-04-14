import streamlit as st
from components.inpaint_gemini_tab import show_inpaint_gemini_tab
from components.inpaint_replicate_tab import show_inpaint_replicate_tab
import os


gemini_api_key = os.environ.get("GEMINI_API_KEY")

s_options = [
    ('gemini', 'Gemini'),
    ('controlnet', 'ControlNet')
]

def on_change_sub_tab():
    st.session_state.current_sub_tab = st.session_state.sub_tab_select[0]

def show_inpaint_tab():
    """インペイントタブの内容を表示する"""
    st.header("インペイント")

    # サブタブの状態を管理
    if 'current_sub_tab' not in st.session_state:
        st.session_state.current_sub_tab = 'gemini'

    # 現在のサブタブに一致するオプションのインデックスを取得
    current_index = next(i for i, (key, _) in enumerate(s_options) if key == st.session_state.current_sub_tab)

    current_sub_tab = st.selectbox(
        label='モデル',
        options=s_options,
        format_func=lambda x: x[1],
        key='sub_tab_select',
        on_change=on_change_sub_tab,
        index=current_index
    )

    if st.session_state.current_sub_tab == 'gemini':
        show_inpaint_gemini_tab()
    elif st.session_state.current_sub_tab == 'controlnet':
        show_inpaint_replicate_tab()
