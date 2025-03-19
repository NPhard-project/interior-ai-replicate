import streamlit as st
from PIL import Image
import io
import base64
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# Gemini APIキーを環境変数から取得
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
else:
    st.warning("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

st.title('Gemini 画像編集')
st.text('画像をアップロードして、Gemini APIで画像編集を行います')

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

# # サイドバーに設定を追加
# st.sidebar.header('設定')

# 画像アップロード
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
    
    # 確認ボタン
    confirm_btn = st.button('送信')
    
    if confirm_btn and prompt and gemini_api_key:
        with st.spinner('Gemini APIに送信中...'):
            try:
                # API キーを明示的に指定してクライアントを初期化
                client = genai.Client(api_key=gemini_api_key)
                
                # 画像をバイナリデータに変換
                # image_data = convert_to_bytes(image)
                
                # Gemini APIに送信
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=[
                        prompt,
                        image,
                        # {"mime_type": "image/png", "data": image_bytes}
                        ],
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image']
                    )
                )

                print(response)
                
                # レスポンスから画像を取得
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                    # 生成された画像を表示
                                    result_image_bytes = base64.b64decode(part.inline_data.data)
                                    result_image = Image.open(io.BytesIO(result_image_bytes))
                                    
                                    # 比較表示
                                    compare_col1, compare_col2 = st.columns(2)
                                    with compare_col1:
                                        st.image(
                                            image,
                                            caption='元の画像',
                                            use_column_width=True,
                                        )
                                    with compare_col2:
                                        st.image(
                                            result_image,
                                            caption='生成結果',
                                            use_column_width=True,
                                        )
                                    
                                    # 生成画像のダウンロードリンク
                                    buffered = io.BytesIO()
                                    result_image.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode()
                                    href = f'<a href="data:file/png;base64,{img_str}" download="gemini_generated.png">生成画像をダウンロード</a>'
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
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f'エラーが発生しました: {str(e)}')
    
    elif confirm_btn and not prompt:
        st.warning("プロンプトを入力してください。")
    
    elif confirm_btn and not gemini_api_key:
        st.error("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
