import streamlit as st
from app.utils.file_handlers import save_uploaded_file

def pdf_uploader():
    uploaded_file = st.file_uploader("{PDF파일을 선택하세요}", type="pdf")
    if uploaded_file is not None:
        st.text("파일 업로드가 되었습니다")
        save_uploaded_file(uploaded_file)
    return uploaded_file