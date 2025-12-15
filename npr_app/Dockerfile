FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update

WORKDIR /npr_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
