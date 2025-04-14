import io
import base64

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