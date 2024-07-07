import os
from app.config import PDF_DIRECTORY

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path