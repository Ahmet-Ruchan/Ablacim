import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from pathlib import Path

load_dotenv()

# PDF klasörü - buraya PDF'lerini koy
PDF_FOLDER = "./docs"


@st.cache_resource
def load_and_process_pdfs():
    """Başlangıçta tüm PDF'leri yükle ve işle"""

    # Klasör yoksa oluştur
    Path(PDF_FOLDER).mkdir(exist_ok=True)

    # PDF dosyalarını bul
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

    if not pdf_files:
        return None, "docs klasöründe PDF bulunamadı!"

    # Metinleri çıkar
    text = ""
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        return None, "PDF'lerden metin çıkarılamadı!"

    # Chunk'la
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Vektör DB oluştur
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Chain oluştur
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain, f"✅ {len(pdf_files)} PDF, {len(chunks)} chunk yüklendi"


def main():
    st.set_page_config(page_title="PDF RAG", page_icon="📚")
    st.header("📚 PDF Chatbot")

    # PDF'leri yükle (cache'lenir, sadece 1 kere çalışır)
    chain, message = load_and_process_pdfs()

    st.sidebar.info(message)
    st.sidebar.markdown(f"**PDF Klasörü:** `{PDF_FOLDER}`")

    if chain is None:
        st.warning(f"'{PDF_FOLDER}' klasörüne PDF dosyalarını koy ve sayfayı yenile.")
        return

    # Chat geçmişi
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Geçmişi göster
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Soru sor
    if question := st.chat_input("Soru sor..."):
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                response = chain({"question": question})
                answer = response["answer"]
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()