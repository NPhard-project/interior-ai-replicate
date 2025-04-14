from PIL.ImageFile import ImageFile
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import GaussianBlur
import torch
import replicate
import io
import base64
import requests
import os
from dotenv import load_dotenv

# コンポーネントのインポート
from components.generate_tab import show_generate_tab
from components.replace_tab import show_replace_tab
from components.inpaint_tab import show_inpaint_tab



options = [
    ('generate', '生成'),
    ('replace', '置換'),
    ('inpaint', 'インペイント')
]

def on_change_tab():
    st.session_state.current_tab = st.session_state.tab_select[0]


# タブの状態を管理するためのセッション状態
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'generate'

# サイドバーの表示
with st.sidebar:
    current_select = st.selectbox(
        label='Method',
        options=options,
        format_func=lambda x: x[1],
        key='tab_select',
        on_change=on_change_tab,
    )

    # st.header('設定')

    # 現在のタブに応じてサイドバーの内容を変更
    # current_tab_config = TAB_CONFIG[st.session_state.current_tab]
    # st.subheader(current_tab_config['sidebar_title'])

    # if st.session_state.current_tab == 'generate':
    #     steps = st.slider('ステップ数', 10, 150, 25, key='sidebar_generate_steps')
    #     cfg = st.slider('CFGスケール', 1.0, 20.0, 7.5, key='sidebar_generate_cfg')

    # elif st.session_state.current_tab == 'replace':
        # r_steps = st.slider('ステップ数', 10, 150, 20)
        # r_scale = st.slider('ガイダンススケール', 0.1, 30.0, 9.0, help='高い値にすると、プロンプトの影響が強まり、意図した画像に近づきやすくなります。  \n  低い値にすると、ランダム性が増して自由な画像が生成されます。')
        # r_value_threshold = st.slider('value threshold', 0.01, 2.00, 0.10, help='低い値にすると細かいエッジも検出されますが、ノイズが増える可能性があります。  \n  高い値にすると強いエッジのみが検出され、細かいディテールが失われることがあります。')
        # r_distance_threshold = st.slider('distance threshold', 0.01, 20.0, 0.10, help='低い値にすると短い線分が多く検出されますが、分割されすぎる可能性があります。  \n  高い値にすると近くの線分が統合されやすくなり、よりシンプルな結果になります。')
        # r_eta = st.slider('eta', 0.0, 1.0, 0.0, help="ノイズ除去の影響度:  \n  0.0→決定論的  \n  1.0→ランダム性増")
        # r_image_resolution = st.select_slider('出力解像度', ['256', '512', '768', '1024'], '512')
        # r_detect_resolution = st.slider('検出強度', 128, 1024, 512)
        # r_num_samples = st.sidebar.select_slider('生成画像数', ['1', '4'], '1')
        # seed_col1, seed_col2 = st.columns([1, 2])
        # with seed_col1:
        #     r_use_random_seed = st.checkbox('ランダムシード', value=True, key='sidebar_replace_random_seed')
        # with seed_col2:
        #     r_seed = st.number_input(
        #         'シード値', 0, 4294967295, disabled=r_use_random_seed
        #     )

    # elif st.session_state.current_tab == 'inpaint':
    #     # サブタブの状態を確認
    #     if 'current_sub_tab' not in st.session_state:
    #         st.session_state.current_sub_tab = 'gemini'

    #     if st.session_state.current_sub_tab == 'gemini':
    #         # Gemini用の設定
    #         pass
    #         # steps = st.slider('ステップ数', 10, 150, 25, key='sidebar_inpaint_gemini_steps')
    #         # cfg = st.slider('CFGスケール', 1.0, 20.0, 7.5, key='sidebar_inpaint_gemini_cfg')

    #     elif st.session_state.current_sub_tab == 'controlnet':
    #         # ControlNet用の設定
    #         i_steps = st.slider('ステップ数', 10, 150, 20, key='sidebar_inpaint_controlnet_steps')
    #         i_scale = st.slider('CFGスケール', 0.1, 30.0, 9.0, key='sidebar_inpaint_controlnet_scale')
    #         i_use_random_seed = st.checkbox('ランダムシード', value=True, key='sidebar_inpaint_controlnet_random_seed')
    #         seed_col1, seed_col2 = st.columns([1, 2])
    #         with seed_col1:
    #             use_random_seed = st.checkbox('ランダムシード', value=True)
    #         with seed_col2:
    #             seed = st.number_input(
    #                 'シード値', 0, 4294967295, disabled=i_use_random_seed
    #             )
    #         st.subheader('マスク設定')
    #         i_blur_radius = st.slider('ぼかし半径', 1, 31, 11, 2)
    #         i_blur_sigma = st.slider('ぼかし強度', 1, 50, 20)

# タブの内容を表示
if st.session_state.current_tab == 'generate':
    show_generate_tab()
elif st.session_state.current_tab == 'replace':
    show_replace_tab()
elif st.session_state.current_tab == 'inpaint':
    show_inpaint_tab()