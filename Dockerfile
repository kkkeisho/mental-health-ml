
FROM python:3.12-slim

WORKDIR /app

# 1. 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. アプリコードとモデルファイルをコンテナ内にコピー
COPY api ./api
COPY models ./models

# 3. コンテナの中で使うポート番号（FastAPI 用）
EXPOSE 8000

# 4. コンテナ起動時に実行するコマンド
#    api/app.py の app という FastAPI インスタンスを uvicorn で起動
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
