FROM python:3.11-slim

RUN apt-get update
RUN groupadd -r groupNPR && useradd -r -g groupNPR userNPR

WORKDIR /npr_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

USER userNPR

CMD ["streamlit", "run", "app.py"]
# docker run --name my_npr_app -p 8501:8501