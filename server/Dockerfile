FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY classifier.py .
COPY .env .

EXPOSE 5123

ENV PORT=5123

CMD ["python3", "server.py"]