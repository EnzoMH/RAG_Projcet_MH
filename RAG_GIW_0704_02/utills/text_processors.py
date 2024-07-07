from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(data)