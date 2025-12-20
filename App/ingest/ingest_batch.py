import os
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# --- 1. AYARLAR ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME", "YasaaVisionDB")
collection_name = os.getenv("COLLECTION_NAME", "palmistry_knowledge")
index_name = os.getenv("INDEX_NAME", "vector_index")

if not openai_key or not mongo_uri:
    raise ValueError("❌ HATA: .env eksik!")

# Modeller
llm = ChatOpenAI(model="gpt-4o", api_key=openai_key, max_tokens=1000)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)


# --- 2. FONKSİYONLAR ---

def get_vector_store():
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name
    )


def analyze_image_with_gpt(image_bytes):
    """Görseli GPT-4o'ya teknik olarak yorumlatır."""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """
    **ROLE:** Expert Chiromancy (Palmistry) Archivist.
    **TASK:** Analyze this scientific diagram from a palmistry book.
    **INSTRUCTIONS:**
    1. Identify the specific line, mount, or hand shape shown.
    2. Describe length, depth, curvature of lines technically.
    3. Locate Marks (Stars, Crosses) relative to mounts accurately.
    4. Read any labels (A, B, C) if present.
    **OUTPUT:** A single detailed paragraph description. No advice, just facts.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )

    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"❌ Görsel analiz hatası: {e}")
        return "Görsel analiz edilemedi."


def process_pdf(pdf_path):
    """Tek bir PDF dosyasını işler."""
    file_name = os.path.basename(pdf_path)

    if not os.path.exists(pdf_path):
        print(f"❌ Dosya yok: {pdf_path}")
        return

    vector_store = get_vector_store()
    doc = fitz.open(pdf_path)

    print(f"\n📘 KİTAP İŞLENİYOR: '{file_name}' ({len(doc)} sayfa)")

    for page_num, page in enumerate(doc):
        real_page_num = page_num + 1

        # Basit bir ilerleme çubuğu (Her sayfada log basmasın, 10 sayfada bir bassın)
        if real_page_num % 10 == 0:
            print(f"   ⏳ {file_name} -> Sayfa {real_page_num} işleniyor...")

        text_content = page.get_text()

        # Görsel Analizi
        image_list = page.get_images(full=True)
        visual_descriptions = []

        if image_list:
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # 5KB altı görselleri atla
                if len(image_bytes) < 5000:
                    continue

                desc = analyze_image_with_gpt(image_bytes)
                visual_descriptions.append(f"[DIAGRAM ANALYSIS]: {desc}")

        # Birleştirme
        combined_text = f"--- PAGE {real_page_num} START ---\n{text_content}\n"
        if visual_descriptions:
            combined_text += "\n--- VISUAL CONTENTS ---\n" + "\n".join(visual_descriptions)
        combined_text += f"\n--- PAGE {real_page_num} END ---"

        # Kaydetme
        if len(combined_text.strip()) > 50:
            metadata = {"source": file_name, "page": real_page_num, "type": "hybrid_book_page"}
            vector_store.add_texts(texts=[combined_text], metadatas=[metadata])

    print(f"✅ TAMAMLANDI: {file_name} veritabanına yüklendi!")


# --- 3. TOPLU ÇALIŞTIRMA (BATCH PROCESS) ---
if __name__ == "__main__":
    # PDF'lerin olduğu klasör adı
    PDF_KLASORU = "pdf_arsiv"

    # Klasör yoksa uyar
    if not os.path.exists(PDF_KLASORU):
        print(f"❌ '{PDF_KLASORU}' klasörü bulunamadı! Lütfen oluşturup içine PDF atın.")
    else:
        # Klasördeki tüm dosyaları bul
        dosyalar = os.listdir(PDF_KLASORU)
        pdf_dosyalari = [f for f in dosyalar if f.lower().endswith('.pdf')]

        print(f"📂 Klasörde {len(pdf_dosyalari)} adet PDF bulundu.")

        # Sırayla işle
        for pdf in pdf_dosyalari:
            tam_yol = os.path.join(PDF_KLASORU, pdf)
            process_pdf(tam_yol)

        print("\n🎉 TÜM KİTAPLAR BAŞARIYLA İŞLENDİ!")