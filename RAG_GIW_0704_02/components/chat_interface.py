import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.text_processors import split_text
from models.llm import get_llm
from models.vectorstore import get_vectorstore

def chat_interface(uploaded_file):
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = get_vectorstore()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if 'qa_chain' not in st.session_state:
        with st.status("파일 처리 중..."):
            loader = PyPDFLoader(uploaded_file.name)
            data = loader.load()
            all_splits = split_text(data)
            st.session_state.vectorstore = st.session_state.vectorstore.from_documents(
                documents=all_splits,
                embedding=HuggingFaceEmbeddings(model_name='beomi/Llama-3-Open-Ko-8B')
            )
            st.session_state.vectorstore.persist()

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        template = """너는 학습하는 챗봇이야, 내가 하는 질문에 전문적이고 지식적으로 답변하길 바래
        Context: {context}
        History: {history}
        User: {question}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question",
        )

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory,
            }
        )

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)