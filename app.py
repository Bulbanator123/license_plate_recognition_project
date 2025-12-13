import requests
import streamlit as st

st.set_page_config(page_title="Number Plates Recognition", page_icon="favicon.ico")
st.title("Number Plates Recognition")
st.write("Распознование номеров")

uploaded_file = st.file_uploader("Загрузить фотку", type=["jpg", "png", "jpeg", ".mp4", ".mov", ".avi", ".webm", ".giff"])
if uploaded_file is not None:
    if st.button("Отправить"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post("http://127.0.0.1:8000/nplr", files=files)
        st.write(response.json())
        if response.status_code == 200:
            st.success("Файл успешно отправлен!")
        else:
            st.error(f"Ошибка: {response.json()}")