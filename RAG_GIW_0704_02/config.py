import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("", "your_default_token_here")

PDF_DIRECTORY = 'pdfFiles'
VECTOR_DB_DIRECTORY = 'vectorDB'

if not os.path.exists(PDF_DIRECTORY):
    os.makedirs(PDF_DIRECTORY)

if not os.path.exists(VECTOR_DB_DIRECTORY):
    os.makedirs(VECTOR_DB_DIRECTORY)