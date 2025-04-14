import streamlit as st
from PIL import Image
import io
import base64
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import GaussianBlur
import torch


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
else:
    st.warning("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

def show_inpaint_gemini_tab():
    st.header('Gemini Inpaint')
    st.text('画像をアップロードして、Gemini APIで画像編集を行います')

    upload_file = st.file_uploader('画像をアップロードしてください', type=['jpg', 'jpeg', 'png'], key='inpaint_gemini_uploader')

    if upload_file is not None:
        # ファイル名を取得
        input_filename = upload_file.name

        # アップロードされた画像を読み込み
        image_bytes = upload_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        # 画像サイズの制限（必要に応じて）
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)

        # 画像表示と情報
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image=image, caption=f'アップロードされた画像: {input_filename}')
        with col2:
            st.info(f'ファイル名: {input_filename}')
            st.info(f'画像サイズ: {image.width} x {image.height}')
            st.info(f'アスペクト比: {image.width/image.height}')

        st.subheader('マスク作成')
        st.write('黒い部分が置き換えられる領域です。マスクを描画してください。')

        col3, col4 = st.columns(2)
        with col3:
            drawing_tool = st.selectbox('描画ツール', options=['freedraw'])
            brush_size = st.slider('ブラシサイズ', 5, 50, 20)
        with col4:
            blur_radius = st.slider(
                'ぼかし半径', 1, 31, 11, 2
            )
            blur_sigma = st.slider('ぼかし強度', 1, 50, 20)

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

        st.subheader('プロンプト')

        # マスク名の設定（オプション）
        mask_name = st.text_input('マスク名（任意）:', value=f'mask_for_{input_filename}')

        # デフォルトプロンプトテンプレートに画像名とマスク名を含める
        instruction_prompt = """
INSTRUCTION: Please inpaint/edit {input_filename} ONLY in the areas shown as black area in the {mask_name}.png.

DO NOT change any other part of the image outside the masked area.
The masked area should be replaced with: {prompt}

Please generate a realistic and seamless result that integrates perfectly with the rest of the image.
"""
        prompt = st.text_area('画像編集のプロンプトを入力してください',
                             help='どのように画像を編集したいかを詳細に記述してください')

        # 確認ボタン
        confirm_btn = st.button('送信')

        if confirm_btn and prompt and gemini_api_key:

            with st.spinner('Gemini APIに送信中...'):
                try:
                    # マスクの処理
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
                                caption=f'作成されたマスク: {mask_name}',
                                use_column_width=True,
                            )

                        mask_tensor = (
                            torch.from_numpy(mask_array / 255.0).float().unsqueeze(0)
                        )
                        blur = GaussianBlur(blur_radius, blur_sigma)
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

                        blurred_mask.save(io.BytesIO(), format="PNG")

                        client = genai.Client(api_key=gemini_api_key)
                        print(instruction_prompt.format(input_filename=input_filename, mask_name=mask_name, prompt=prompt))
                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp-image-generation",
                            contents=[
                                instruction_prompt.format(input_filename=input_filename, mask_name=mask_name, prompt=prompt),
                                image,
                                blurred_mask
                            ],
                            config=types.GenerateContentConfig(
                                response_modalities=['Text', 'Image']
                            )
                        )

                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'content') and candidate.content:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                            result_image_bytes = base64.b64decode(part.inline_data.data)
                                            result_image = Image.open(io.BytesIO(result_image_bytes))

                                            # 結果ファイル名の生成
                                            result_filename = f"edited_{input_filename}"

                                            # 比較表示
                                            compare_col1, compare_col2 = st.columns(2)
                                            with compare_col1:
                                                st.image(
                                                    image,
                                                    caption=f'元の画像: {input_filename}',
                                                    use_column_width=True,
                                                )
                                            with compare_col2:
                                                st.image(
                                                    result_image,
                                                    caption=f'生成結果: {result_filename}',
                                                    use_column_width=True,
                                                )

                                            # 生成画像のダウンロードリンク
                                            buffered = io.BytesIO()
                                            result_image.save(buffered, format="PNG")
                                            img_str = base64.b64encode(buffered.getvalue()).decode()
                                            href = f'<a href="data:file/png;base64,{img_str}" download="{result_filename}">生成画像をダウンロード</a>'
                                            st.markdown(href, unsafe_allow_html=True)

                        # テキストレスポンスがあれば表示
                        text_response = ""
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'content') and candidate.content:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text_response += part.text

                        if text_response:
                            st.subheader("Geminiからのレスポンス")
                            st.write(text_response)
                    else:
                        st.error('マスクが作成されていません。キャンバスに描画してください')

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    st.error(f'エラーが発生しました: {str(e)}')

        elif confirm_btn and not prompt:
            st.warning("プロンプトを入力してください。")

        elif confirm_btn and not gemini_api_key:
            st.error("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
