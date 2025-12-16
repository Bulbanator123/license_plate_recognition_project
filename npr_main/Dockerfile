FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /npr_main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm -rf /usr/local/cuda* \
    /usr/lib/x86_64-linux-gnu/libcuda* \
    /usr/lib/x86_64-linux-gnu/libnvidia*

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
# docker run --name my_npr_app -p 127.0.0.1:8000:8000
