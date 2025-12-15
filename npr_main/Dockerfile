FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /npr_main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service:app", "--reload"]
