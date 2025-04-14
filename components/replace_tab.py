import streamlit as st
from PIL import Image
import io
import os
from dotenv import load_dotenv
import numpy as np
import requests
import replicate

from components.utils.image_handler import image_to_base64

load_dotenv()

replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_api_token:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
else:
    st.warning("REPLICATE_API_TOKEN が設定されていません。.env ファイルを確認してください。")

def show_replace_tab():
    """置換タブの内容を表示する"""
    st.header('画像置換')
    st.text('画像の特定の領域を新しいコンテンツで置き換えるデモです')

    # サイドバーの設定
    with st.sidebar:
        if st.session_state.current_tab == 'replace':
            r_steps = st.slider('ステップ数', 10, 150, 20)
            r_scale = st.slider('ガイダンススケール', 0.1, 30.0, 9.0, help='高い値にすると、プロンプトの影響が強まり、意図した画像に近づきやすくなります。  \n  低い値にすると、ランダム性が増して自由な画像が生成されます。')
            r_value_threshold = st.slider('value threshold', 0.01, 2.00, 0.10, help='低い値にすると細かいエッジも検出されますが、ノイズが増える可能性があります。  \n  高い値にすると強いエッジのみが検出され、細かいディテールが失われることがあります。')
            r_distance_threshold = st.slider('distance threshold', 0.01, 20.0, 0.10, help='低い値にすると短い線分が多く検出されますが、分割されすぎる可能性があります。  \n  高い値にすると近くの線分が統合されやすくなり、よりシンプルな結果になります。')
            r_eta = st.slider('eta', 0.0, 1.0, 0.0, help="ノイズ除去の影響度:  \n  0.0→決定論的  \n  1.0→ランダム性増")
            r_image_resolution = st.select_slider('出力解像度', ['256', '512', '768', '1024'], '512')
            r_detect_resolution = st.slider('検出強度', 128, 1024, 512)
            r_num_samples = st.select_slider('生成画像数', ['1', '4'], '1')

            seed_col1, seed_col2 = st.columns([1, 2])
            with seed_col1:
                r_use_random_seed = st.checkbox('ランダムシード', value=True)
            with seed_col2:
                r_seed = st.number_input(
                    'シード値', 0, 4294967295, disabled=r_use_random_seed
                )

    upload_file = st.file_uploader(label='画像をアップロードしてください', type=['jpg', 'jpeg', 'png'], key='replace_uploader')
    if upload_file is not None:
        image_bytes = upload_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image=image, caption='アップロードされた画像')
        with col2:
            st.info(f'画像サイズ: {image.width} x {image.height}')
            st.info(f'アスペクト比: {image.width/image.height}')

        st.subheader('生成プロンプト')
        prompt = st.text_area('生成プロンプト')
        n_prompt = st.text_area('ネガティブプロンプト')
        generate_btn = st.button('画像生成')
        if generate_btn and prompt:
            with st.spinner('画像生成中'):
                try:
                    actual_seed = (
                        np.random.randint(0, 2**32 - 1) if r_use_random_seed else r_seed
                    )
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text('APIリクエスト送信中...')
                    progress_bar.progress(10)
                    request_params = {
                        "eta": r_eta,
                        "seed": actual_seed,
                        "image": "data:image/png;base64," + image_to_base64(image=image),
                        "scale": r_scale,
                        "prompt": prompt,
                        "a_prompt": "best quality, high resolution, extremely detailed",
                        "n_prompt": n_prompt,
                        "ddim_steps": r_steps,
                        "num_samples": r_num_samples,
                        "value_threshold": r_value_threshold,
                        "image_resolution": r_image_resolution,
                        "detect_resolution": r_detect_resolution,
                        "distance_threshold": r_distance_threshold,
                    }
                    output = replicate.run(
                        "jagilley/controlnet-hough:854e8727697a057c525cdb45ab037f64ecca770a1769cc52287c2e56472a247b",
                        input=request_params,
                    )
                    if output and len(output) > 0:
                        st.subheader('元の画像')
                        st.image(image, caption='元の画像', use_column_width=True)
                        result_images = []
                        for result_image_url in output:
                            response = requests.get(result_image_url, stream=True)
                            if response.status_code == 200:
                                result_image = Image.open(io.BytesIO(response.content))
                                result_images.append(result_image)
                        st.subheader('生成結果')
                        if len(result_images) > 0:
                            num_cols = 2
                            cols = st.columns(num_cols)
                            for i, result_image in enumerate(result_images):
                                col_idx = i % num_cols
                                cols[col_idx].image(result_image, caption=f'生成結果 #{i+1}', use_column_width=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    st.error(f'エラーが発生しました: {str(e)}')

        elif generate_btn and not prompt:
            st.warning('プロンプトを入力してください')
