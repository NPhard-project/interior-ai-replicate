import streamlit as st
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import GaussianBlur
import torch
import requests
import replicate

from components.utils.image_handler import image_to_base64

load_dotenv()

replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_api_token:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
else:
    st.warning("REPLICATE_API_TOKEN が設定されていません。.env ファイルを確認してください。")

def show_inpaint_replicate_tab():
    """ControlNetを使用したインペイントタブの内容を表示する"""
    st.header("ControlNet インペイント")

    # # サイドバーの設定を取得
    # i_steps = st.session_state.get('sidebar_inpaint_controlnet_steps', 20)
    # i_scale = st.session_state.get('sidebar_inpaint_controlnet_scale', 9.0)
    # i_use_random_seed = st.session_state.get('sidebar_inpaint_controlnet_random_seed', True)
    # i_blur_radius = st.session_state.get('sidebar_inpaint_controlnet_blur_radius', 11)
    # i_blur_sigma = st.session_state.get('sidebar_inpaint_controlnet_blur_sigma', 20)

    with st.sidebar:
        if st.session_state.current_tab == 'inpaint':
            if st.session_state.current_sub_tab == 'controlnet':
                i_steps = st.slider('ステップ数', 10, 150, 20)
                i_scale = st.slider('ガイダンススケール', 0.1, 30.0, 9.0, help='高い値にすると、プロンプトの影響が強まり、意図した画像に近づきやすくなります。  \n  低い値にすると、ランダム性が増して自由な画像が生成されます。')
                # i_use_random_seed = st.checkbox('ランダムシード', value=True)
                i_blur_radius = st.slider('ぼかし半径', 1, 31, 11, 2)
                i_blur_sigma = st.slider('ぼかし強度', 1, 50, 20)

    # ファイルアップローダー
    upload_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])

    if upload_file is not None:
        image_bytes = upload_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        # 画像サイズの制限
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)

        # 画像表示
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image=image, caption='アップロードされた画像')
        with col2:
            st.info(f'画像サイズ: {image.width} x {image.height}')
            st.info(f'アスペクト比: {image.width/image.height}')

        # マスク作成
        st.subheader('マスク作成')
        st.write('黒い部分が置き換えられる領域です。マスクを描画してください。')

        col3, col4 = st.columns(2)
        with col3:
            brush_size = st.slider('ブラシサイズ', 5, 50, 20)
        with col4:
            drawing_tool = st.selectbox('描画ツール', options=['freedraw'])

        container_width = 700
        aspect_ratio = image.height / image.width
        canvas_width = min(container_width, image.width)
        canvas_height = int(canvas_width * aspect_ratio)

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=brush_size,
            stroke_color="black",
            background_color="white",
            background_image=image.resize((canvas_width, canvas_height)),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_tool,
            key="canvas",
            display_toolbar=False,
        )

        st.subheader('生成プロンプト')
        prompt = st.text_area('生成プロンプト')
        negative_prompt = st.text_area('ネガティブプロンプト(生成から除外したい要素)')
        generate_btn = st.button('画像生成')
        if generate_btn and prompt:
            with st.spinner('画像生成中'):
                try:
                    if canvas_result.image_data is not None:
                        mask_data = canvas_result.image_data
                        mask_image_resized = Image.fromarray(
                            mask_data.astype(np.uint8)
                        ).resize(image.size)
                        mask_data_resized = np.array(mask_image_resized)

                        mask_array = (
                            np.ones(
                                (mask_data_resized.shape[0], mask_data_resized.shape[1]),
                                dtype=np.uint8,
                            )
                            * 255
                        )

                        black_pixels = (
                            (mask_data_resized[:, :, 0] < 10)
                            & (mask_data_resized[:, :, 1] < 10)
                            & (mask_data_resized[:, :, 2] < 10)
                            & (mask_data_resized[:, :, 3] > 0)
                        )
                        mask_array[black_pixels] = 0

                        mask_image = Image.fromarray(mask_array)

                        mask_col1, mask_col2 = st.columns(2)
                        with mask_col1:
                            st.image(
                                mask_image,
                                caption='作成されたマスク',
                                use_column_width=True,
                            )

                        mask_tensor = (
                            torch.from_numpy(mask_array / 255.0).float().unsqueeze(0)
                        )
                        blur = GaussianBlur(i_blur_radius, i_blur_sigma)
                        blurred_mask_tensor = blur(mask_tensor)
                        blurred_mask = Image.fromarray(
                            (blurred_mask_tensor.squeeze().numpy() * 255).astype(np.uint8)
                        )
                        with mask_col2:
                            st.image(
                                blurred_mask,
                                caption='ぼかし適用後のマスク',
                                use_column_width=True,
                            )

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text('APIリクエスト送信中...')
                        progress_bar.progress(10)

                        def get_valid_size(size):
                            valid_sizes = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
                            return min(valid_sizes, key=lambda x: abs(x - size))

                        mask_array = 255 - mask_array

                        request_params = {
                            "mask": "data:image/png;base64," + image_to_base64(Image.fromarray(mask_array)),
                            "image": "data:image/png;base64," + image_to_base64(image),
                            "width": get_valid_size(image.width),
                            "height": get_valid_size(image.height),
                            "prompt": prompt,
                            "scheduler": "DPMSolverMultistep",
                            "num_outputs": 1,
                            "guidance_scale": i_scale,
                            "num_inference_steps": i_steps,
                        }

                        if negative_prompt:
                            request_params['negative_prompt'] = negative_prompt

                        status_text.text('画像生成中...')
                        progress_bar.progress(30)

                        answers = replicate.run(
                            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                            input=request_params
                        )

                        if answers and len(answers) > 0:
                            result_image_url = answers[0]

                            # URLから画像をダウンロード
                            response = requests.get(result_image_url, stream=True)
                            if response.status_code == 200:
                                result_image = Image.open(io.BytesIO(response.content))

                                compare_col1, compare_col2 = st.columns(2)
                                with compare_col1:
                                    st.image(image, caption='元の画像', use_column_width=True)
                                with compare_col2:
                                    st.image(result_image, caption='生成結果', use_column_width=True)
                            else:
                                st.error(f'画像のダウンロードに失敗しました: {response.status_code}')
                    else:
                        st.error('マスクが作成されていません。キャンバスに描画してください')

                except Exception as e:
                    st.error(f'エラーが発生しました: {str(e)}')
