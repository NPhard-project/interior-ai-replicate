# Interior AI
## 準備
### Replicate アカウント設定
[Replicate](https://replicate.com/home) からアカウント登録 / Billing の設定をする(有償)
### pip install requirements.txt
`pip install requirements.txt` コマンドを実行
### python 環境の設定
python のバージョンは 3.x で新しめのものならなんでも問題ないと思う
確認環境は 3.12.7
python のバージョン管理を pyenv でしたい場合は[このあたり](https://shingetsutan.net/centos-django-deploy/#toc_id_2) 見てもらえると
もしくは poetry でバージョン管理してもよき
## 実行
### ControlNet 版の実行
`streamlit run controlnet.py`
### Inpaint 版の実行
`streamlit run inpaint.py`
## ブラウザで起動
`localhost:8501` にアクセス