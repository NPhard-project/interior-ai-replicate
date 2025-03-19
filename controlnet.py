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

# .env ファイルから環境変数を読み込む
load_dotenv()

# レプリケートのAPIトークンを環境変数から取得
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_api_token:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
else:
    st.warning("REPLICATE_API_TOKEN が設定されていません。.env ファイルを確認してください。")

def convert_to_bytes(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    return base64.b64encode(byte_stream.getvalue()).decode('utf-8')

st.title('ControlNet demo')
st.text('画像をもとに要素の追加、変更を行うデモです')
st.text('画像をアップロードしてください')
st.sidebar.header('設定')
basic_tab, advanced_tab = st.sidebar.tabs(['基本設定', '詳細設定'])
with basic_tab:
    steps = st.slider('ステップ数', 10, 150, 20)
    scale = st.slider('ガイダンススケール', 0.1, 30.0, 9.0, help='高い値にすると、プロンプトの影響が強まり、意図した画像に近づきやすくなります。  \n  低い値にすると、ランダム性が増して自由な画像が生成されます。')
    value_threshold = st.slider('value threshold', 0.01, 2.00, 0.10, help='低い値にすると細かいエッジも検出されますが、ノイズが増える可能性があります。  \n  高い値にすると強いエッジのみが検出され、細かいディテールが失われることがあります。')
    distance_threshold = st.slider('distance threshold', 0.01, 20.0, 0.10, help='低い値にすると短い線分が多く検出されますが、分割されすぎる可能性があります。  \n  高い値にすると近くの線分が統合されやすくなり、よりシンプルな結果になります。')
    eta = st.slider('eta', 0.0, 1.0, 0.0, help="ノイズ除去の影響度:  \n  0.0→決定論的  \n  1.0→ランダム性増")
    image_resolution = st.select_slider('出力解像度', ['256', '512', '768', '1024'], '512')
    detect_resolution = st.slider('検出強度', 128, 1024, 512)
    num_samples = st.gsidebar.select_slider('生成画像数', ['1', '4'], '1') 
    
with advanced_tab:
    seed_col1, seed_col2 = st.columns([1, 2])
    with seed_col1:
        use_random_seed = st.checkbox('ランダムシード', value=True)
    with seed_col2:
        seed = st.number_input(
            'シード値', 0, 4294967295, disabled=use_random_seed
        )
    start_schedule = st.slider(
        'スタートスケジュール',
        0.0,
        1.0,
        1.0,
        0.01
    )

upload_file = st.file_uploader('Drag and drop file here')
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
                    np.random.randint(0, 2**32 - 1) if use_random_seed else seed
                )
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text('APIリクエスト送信中...')
                progress_bar.progress(10)
                request_params = {
                    "eta": eta,
                    "seed": actual_seed,
                    "image": "data:image/png;base64," + convert_to_bytes(image=image),
                    "scale": scale,
                    "prompt": prompt,
                    "a_prompt": "best quality, extremely detailed",
                    "n_prompt": n_prompt,
                    "ddim_steps": steps,
                    "num_samples": num_samples,
                    "value_threshold": value_threshold,
                    "image_resolution": image_resolution,
                    "detect_resolution": detect_resolution,
                    "distance_threshold": distance_threshold,
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
