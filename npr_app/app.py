import requests
import streamlit as st
import json
from io import BytesIO
from PIL import Image


st.set_page_config(page_title="Number Plates Recognition", page_icon="favicon.ico")
st.title("Number Plates Recognition")
st.write("Распознование номеров")

uploaded_file = st.file_uploader("Загрузить фотку", type=["jpg", "png", ".mp4"])
if uploaded_file is not None:
    if st.button("Отправить"):
        col1, col2 = st.columns(2)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        det_response = requests.post("http://127.0.0.1:8000/nplr", files=files)
        st.write(det_response.json())
        if det_response.status_code == 200:
            st.success("Файл успешно отправлен!")
        else:
            st.error(f"Ошибка: {det_response.json()}")
        data = det_response.json()
        image_response = requests.post("http://127.0.0.1:8000/nplr/image", data={"data": json.dumps(data)}, files=files)
        print(image_response)
        if image_response.status_code == 200:
            if uploaded_file.type in ("image/png", "image/jpeg"):
                img = Image.open(BytesIO(image_response.content))
                with col1:
                    st.image(uploaded_file)
                with col2:
                    st.image(img)
            elif uploaded_file.type in ("video/mp4",):
                    video_bytes = image_response.content 

                    with col1:
                        st.video(uploaded_file)

                    with col2:
                        st.video(video_bytes)
                        # st.download_button(
                        #     label="Скачать видео",
                        #     data=video_bytes,
                        #     file_name="result.mp4",
                        #     mime="video/mp4"
                        # )