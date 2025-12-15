FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg
RUN groupadd -r groupNPR && useradd -r -g groupNPR userNPR

WORKDIR /npr_main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

USER userNPR

CMD ["uvicorn", "service:app", "--reload"]
# docker run --name my_npr_app -p 127.0.0.1:8000:8000