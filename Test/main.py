import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# .env dosyasından API key'i yükle
load_dotenv()


def get_pdf_text(pdf_files):
    """PDF dosyalarından metin çıkar"""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(text):
    """Metni chunk'lara böl"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    """ChromaDB vektör veritabanı oluştur"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    """Konuşma zinciri oluştur"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain


def main():
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚")
    st.header("📚 PDF ile Sohbet Et")

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar - PDF yükleme
    with st.sidebar:
        st.subheader("📁 PDF Yükle")
        pdf_files = st.file_uploader(
            "PDF dosyalarını seç",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("İşle", type="primary"):
            if pdf_files:
                with st.spinner("PDF'ler işleniyor..."):
                    # Metin çıkar
                    raw_text = get_pdf_text(pdf_files)

                    if not raw_text.strip():
                        st.error("PDF'lerden metin çıkarılamadı!")
                        return

                    # Chunk'la
                    chunks = get_text_chunks(raw_text)
                    st.info(f"✅ {len(chunks)} chunk oluşturuldu")

                    # Vektör DB oluştur
                    vectorstore = get_vectorstore(chunks)

                    # Konuşma zinciri
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Hazır! Soru sorabilirsin.")
            else:
                st.warning("Önce PDF yükle!")

    # Chat arayüzü
    if st.session_state.conversation:
        # Sohbet geçmişini göster
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Kullanıcı sorusu
        user_question = st.chat_input("PDF hakkında bir soru sor...")

        if user_question:
            # Kullanıcı mesajını ekle
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.write(user_question)

            # Cevap al
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    response = st.session_state.conversation({
                        "question": user_question
                    })
                    answer = response["answer"]
                    st.write(answer)

            # Asistan cevabını ekle
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
    else:
        st.info("👈 Başlamak için sol panelden PDF yükle ve 'İşle' butonuna bas.")


if __name__ == "__main__":
    main()