from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import VECTOR_DB_DIRECTORY

def get_vectorstore():
    return Chroma(
        persist_directory=VECTOR_DB_DIRECTORY,
        embedding_function=HuggingFaceEmbeddings(model_name='beomi/Llama-3-Open-Ko-8B')
    )