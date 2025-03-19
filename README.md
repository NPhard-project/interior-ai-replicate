# Interior AI (ControlNet / Inpaint)

## 準備

### Replicate アカウント設定
[Replicate](https://replicate.com/home) からアカウント登録 / Billing の設定をする(有償)

#### 環境変数に API キーを設定
root directory 直下の `.env` ファイル(なければ作成)に以下記述を追加
```
REPLICATE_API_TOKEN=r8_NXp**********************************
```

### pip install requirements.txt
`pip install requirements.txt` コマンドを実行

### python 環境の設定

python のバージョンは 3.x で新しめのものならなんでも問題ないと思う

確認環境は 3.12.7

python のバージョン管理を pyenv でしたい場合は[このあたり](https://shingetsutan.net/centos-django-deploy/#toc_id_2) 見てもらえると

もしくは poetry でバージョン管理してもよき

## 実行
コンソールで以下コマンドを実行
### ControlNet 版の実行
```
streamlit run controlnet.py
```

### Inpaint 版の実行
```
streamlit run inpaint.py
```

## ブラウザで起動
`localhost:8501` にアクセス


# Gemini.py の利用準備

## 準備
### pip install requirements.txt
`pip install requirements.txt` コマンドを実行
<!-- ### google-genai のインストール
python 3.9 以降を利用して以下コマンドを実行
`pip install google-genai` -->

### API キーを設定

#### Gemini API キーを取得
[Google AI Studio](https://aistudio.google.com/app/apikey?hl=ja)

#### 環境変数に API キーを設定
root directory 直下の `.env` ファイルに以下記述を追加

```
GEMINI_API_KEY=<YOUR_API_KEY_HERE>
```

## 実行
コンソールで以下コマンドを実行
```
streamlit run gemini.py
```