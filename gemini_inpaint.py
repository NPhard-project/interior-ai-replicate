import streamlit as st
from PIL import Image
import io
import base64
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import numpy as np
import torch
from torchvision.transforms import GaussianBlur
from streamlit_drawable_canvas import st_canvas
import requests
import traceback

# .env ファイルから環境変数を読み込む
load_dotenv()

# セッション状態の初期化
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'  # 初期ステージ: 'upload', 'generate', 'inpaint'
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

def set_stage(stage):
    st.session_state.stage = stage

# Gemini APIキーを環境変数から取得
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
else:
    st.warning("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

# 画像をバイトに変換する関数
def convert_to_bytes(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    return byte_stream.getvalue()

# 画像をbase64エンコードする関数
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

st.title('Gemini Inpaint 画像編集')
st.text('画像をアップロードして、Gemini APIで画像編集を行います')

# 画像アップロード

if st.session_state.stage == 'upload':
    upload_file = st.file_uploader('画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
    if upload_file is not None:
        # アップロードされた画像を読み込み
        image_bytes = upload_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 画像サイズの制限（必要に応じて）
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
            
        st.session_state.original_image = image  # session_state に保存
        
        # 画像表示と情報
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image=image, caption='アップロードされた画像')
        with col2:
            st.info(f'画像サイズ: {image.width} x {image.height}')
            st.info(f'アスペクト比: {image.width/image.height}')
        
        # テキストプロンプト入力
        st.subheader('プロンプト')
        prompt = st.text_area('画像編集のプロンプトを入力してください', 
                            help='どのように画像を編集したいかを詳細に記述してください')
        
        # 確認ボタン - ここでステートを変更するだけ
        if st.button('画像生成'):
            if prompt:
                st.session_state.prompt = prompt
                set_stage('generate')
                st.rerun()
            else:
                st.warning("プロンプトを入力してください。")

elif st.session_state.stage == 'generate':
    # 戻るボタン
    if st.button('戻る', key='back_to_upload'):
        set_stage('upload')
        st.rerun()
    
    # 元の画像とプロンプトの表示
    st.subheader('元の画像')
    st.image(st.session_state.original_image, caption='元の画像')
    st.info(f"プロンプト: {st.session_state.prompt}")
    
    # 画像がまだ生成されていない場合のみリクエストを実行
    if st.session_state.result_image is None:
        with st.spinner('Gemini APIに送信中...'):
            try:
                # API キーを明示的に指定してクライアントを初期化
                client = genai.Client(api_key=gemini_api_key)
                
                # Gemini APIに送信
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=[
                        st.session_state.prompt,
                        st.session_state.original_image,
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image']
                    )
                )

                # レスポンスから画像とテキストを取得
                result_image = None
                text_response = ""
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                    result_image_bytes = base64.b64decode(part.inline_data.data)
                                    result_image = Image.open(io.BytesIO(result_image_bytes))
                                if hasattr(part, 'text') and part.text:
                                    text_response += part.text
                
                # 生成結果をセッションに保存
                if result_image:
                    st.session_state.result_image = result_image
                    st.session_state.text_response = text_response
                else:
                    st.error('画像生成に失敗しました。')
            except Exception as e:
                traceback.print_exc()
                st.error(f'エラーが発生しました: {str(e)}')
    
    # 生成された画像の表示 (セッションから取得)
    if st.session_state.result_image:
        st.subheader('生成結果')
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            st.image(
                st.session_state.original_image,
                caption='元の画像',
                use_column_width=True,
            )
        with compare_col2:
            st.image(
                st.session_state.result_image,
                caption='生成結果',
                use_column_width=True,
            )
        
        # 生成画像のダウンロードリンク
        buffered = io.BytesIO()
        st.session_state.result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="gemini_generated.png">生成画像をダウンロード</a>'
        st.markdown(href, unsafe_allow_html=True)

        # テキストレスポンスの表示
        if hasattr(st.session_state, 'text_response') and st.session_state.text_response:
            st.subheader("Geminiからのレスポンス")
            st.write(st.session_state.text_response)
        
        # 次の操作ボタン
        col3, col4 = st.columns(2)
        with col3:
            if st.button('インペイント', key='to_inpaint'):
                set_stage('inpaint')
                st.rerun()
        with col4:
            if st.button('最初からやり直す', key='reset_from_generate'):
                set_stage('upload')
                # 生成画像をクリア
                st.session_state.result_image = None
                st.session_state.text_response = None
                st.rerun()
elif st.session_state.stage == 'inpaint':
    if st.session_state.result_image is None:
        st.error('生成結果がありません。最初からやり直してください')
        if st.button('最初からやり直す', key='reset_from_inpaint_error'):
            set_stage('upload')
            st.rerun()
    else:
        st.subheader('インペイント (マスク作成)')
        st.write('黒い部分が置き換えられる領域です。マスクを描画してください。')
        
        # 生成画像の表示
        st.image(st.session_state.result_image, caption='生成結果', width=400)
        
        # キャンバス設定
        col5, col6 = st.columns(2)
        with col5:
            brush_size = st.slider('ブラシサイズ', 5, 50, 20)
        with col6:
            drawing_tool = st.selectbox('描画ツール', options=['freedraw'])
        
        # キャンバスサイズ計算
        container_width = 700
        aspect_ratio = st.session_state.result_image.height / st.session_state.result_image.width
        canvas_width = min(container_width, st.session_state.result_image.width)
        canvas_height = int(canvas_width * aspect_ratio)
        
        # マスク作成用のキャンバス
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=brush_size,
            stroke_color="black",
            background_color="white",
            background_image=st.session_state.result_image.resize((canvas_width, canvas_height)),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_tool,
            key="canvas",
            display_toolbar=False,
        )

        st.subheader('マスク設定')
        blur_radius = st.slider('ぼかし半径', 1, 31, 11, 2)
        blur_sigma = st.slider('ぼかし強度', 1, 50, 20)

        st.subheader('インペイントプロンプト')
        inpaint_prompt = st.text_area('選択した領域の生成プロンプト', 
                                    help='マスクした領域をどのように変更するかを記述してください')
        col1, col2, col3 = st.columns(3)
        with col1:
            inpaint_btn = st.button('インペイント実行')
        with col2:
            back_btn = st.button('戻る')
        with col3:
            reset_btn = st.button('リセット')

        if back_btn:
            set_stage('generate')
            st.rerun()

        if reset_btn:
            set_stage('upload')
            st.session_state.result_image = None
            st.rerun()
                        
        if inpaint_btn and inpaint_prompt:
            if canvas_result.image_data is None:
                st.error('マスクが作成されていません。キャンバスに描画してください')
            else:
                with st.spinner('インペイント処理中...'):
                    try:
                        mask_data = canvas_result.image_data
                        mask_image_resized = Image.fromarray(
                            mask_data.astype(np.uint8)
                        ).resize(st.session_state.result_image.size)
                        mask_data_resized = np.array(mask_image_resized)
                        
                        # マスク作成
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
                        
                        # マスクをぼかす
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
                        
                        # バイナリデータをBase64エンコード
                        result_image_b64 = image_to_base64(st.session_state.result_image)

                        if blurred_mask.mode != 'L':
                            blurred_mask = blurred_mask.convert('L')
                        mask_image_b64 = image_to_base64(blurred_mask)

                        st.info('Gemini APIに送信中...')

                        # API クライアントの初期化
                        client = genai.Client(api_key=gemini_api_key)
                        
                        # インペイントリクエストの送信
                        inpaint_response = client.models.generate_content(
                            model="gemini-2.0-flash-exp-image-generation",
                            contents=[
                                inpaint_prompt,
                                st.session_state.result_image,
                                blurred_mask
                            ]
                        )

                        print(inpaint_response)
                        
                        # インペイント結果処理
                        if hasattr(inpaint_response, 'candidates') and inpaint_response.candidates:
                            for candidate in inpaint_response.candidates:
                                if hasattr(candidate, 'content') and candidate.content:
                                    print(candidate.content)
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                            # 生成された画像を表示
                                            inpaint_image_bytes = base64.b64decode(part.inline_data.data)
                                            inpaint_image = Image.open(io.BytesIO(inpaint_image_bytes))
                                            
                                            # 比較表示
                                            inpaint_col1, inpaint_col2 = st.columns(2)
                                            with inpaint_col1:
                                                st.image(
                                                    st.session_state.result_image,
                                                    caption='マスク前の画像',
                                                    use_column_width=True,
                                                )
                                            with inpaint_col2:
                                                st.image(
                                                    inpaint_image,
                                                    caption='インペイント結果',
                                                    use_column_width=True,
                                                )
                                            
                                            # インペイント結果のダウンロードリンク
                                            buffered = io.BytesIO()
                                            inpaint_image.save(buffered, format="PNG")
                                            img_str = base64.b64encode(buffered.getvalue()).decode()
                                            href = f'<a href="data:file/png;base64,{img_str}" download="gemini_inpainted.png">インペイント結果をダウンロード</a>'
                                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.error('インペイント生成に失敗しました。マスクを調整して再試行してください。')
                    except Exception as e:
                        traceback.print_exc()
                        st.error(f'インペイント処理でエラーが発生しました: {str(e)}')
