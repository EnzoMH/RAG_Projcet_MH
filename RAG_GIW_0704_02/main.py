from components.pdf_uploader import pdf_uploader
from models import llm, vectorstore
from utils import file_handlers, text_processors

def main():
    st.title("Chatbot - to talk to PDFs")
    
    uploaded_file = pdf_uploader()
    
    if uploaded_file is not None:
        chat_interface(uploaded_file)
    else:
        st.write("시작하시려면 PDF를 업로드해주세요")

if __name__ == "__main__":
    main()