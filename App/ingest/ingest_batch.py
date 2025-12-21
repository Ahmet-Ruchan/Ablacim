"""
============================================
YASAA VISION - PDF Ingestion Pipeline (v2)
============================================
Bu script, palmistry (el falı) kitaplarını işleyerek:
1. PDF'lerden metin çıkarır
2. Görselleri GPT-4o Vision ile analiz eder
3. Birleştirilmiş veriyi MongoDB Atlas'a vektör olarak kaydeder

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

import os
import logging
import base64
from typing import Optional, List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from marshmallow import missing
from openai import embeddings
from pymongo import MongoClient
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)
from langchain_core.messages import HumanMessage
from langchain_mongodb import MongoDBAtlasVectorSearch


# ============================================
# LOGGING AYARLARI
# ============================================
# Loglama formatını ayarlıyoruz: zaman - seviye - mesaj
logging.basicConfig(
    level=logging.INFO,                        # INFO ve üstü logları göster
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log formatı
    datefmt="%Y-%m-%d %H:%M:%S"                # Tarih formatı
)
logger = logging.getLogger(__name__)           # Bu modül için logger oluştur


# ============================================
# ENVIRONMENT DEĞİŞKENLERİ YÜKLEME
# ============================================
# .env dosyasını yükle (proje kök dizininde olmalı)

load_dotenv()

# --- API Anahtarları ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")      # OpenAI API anahtarı
MONGO_URI: str = os.getenv("MONGO_URI", "")                # MongoDB bağlantı URI'si

# --- MongoDB Ayarları ---
DB_NAME: str = os.getenv("DB_NAME", "YasaaVisionDB")                    # Veritabanı adı
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "palmistry_knowledge")  # Koleksiyon adı
INDEX_NAME: str = os.getenv("INDEX_NAME", "vector_index")               # Vektör index adı

# --- Model Ayarları ---
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")                 # Görsel analiz modeli
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Embedding modeli
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))                  # Maksimum token sayısı

# --- Dosya Ayarları ---
PDF_FOLDER: str = os.getenv("PDF_FOLDER", "pdf_storage")                # PDF klasör yolu
MIN_IMAGE_SIZE: int = int(os.getenv("MIN_IMAGE_SIZE", "3000"))          # Min görsel boyutu (byte)
LOG_INTERVAL: int = int(os.getenv("LOG_INTERVAL", "10"))                # Kaç sayfada bir log basılsın


# ============================================
# DOĞRULAMA - Kritik değişkenler var mı?
# ============================================

def validate_environment() -> bool:

    required_vars = {
        "OPENAI_API_KEY" : OPENAI_API_KEY,
        "MONGO_URI" : MONGO_URI,
    }

    missing = [] # Eksik değişkenleri tutacak liste

    for var_name, var_value in required_vars.items():
        if not var_value:
            missing.append(var_name)

    # Eksik varsa hata logla ve False döndür
    if missing:
        logger.error(f"❌ Eksik environment değişkenleri: {', '.join(missing)}")
        logger.error("💡 Lütfen .env dosyanızı kontrol edin!")
        return False

    logger.info("✅ Environment değişkenleri doğrulandı")
    return True


# ============================================
# MODEL İNİTİALİZASYONU
# ============================================

def initialize_models() -> tuple[ChatOpenAI, OpenAIEmbeddings]:

    logger.info(f"🤖 Modeller yükleniyor: Vision={VISION_MODEL}, Embedding={EMBEDDING_MODEL}")

    # GPT-4o Vision modeli
    llm = ChatOpenAI(
        model=VISION_MODEL,
        api_key=OPENAI_API_KEY,
        max_tokens=MAX_TOKENS
    )

    # Embedding modeli
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    logger.info("✅ Modeller başarıyla yüklendi")
    return llm, embeddings

# ============================================
# MONGODB VECTOR STORE BAĞLANTISI
# ============================================

def get_vector_store(embeddings: OpenAIEmbeddings) -> MongoDBAtlasVectorSearch:

    logger.info(f"🔌 MongoDB'ye bağlanılıyor: {DB_NAME}/{COLLECTION_NAME}")

    # Client oluştur
    client = MongoClient(MONGO_URI)

    # Collection referansı
    collection = client[DB_NAME][COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    logger.info("✅ MongoDB bağlantısı başarılı")
    return vector_store


# ============================================
# GÖRSEL ANALİZ FONKSİYONU (GPT-4o Vision)
# ============================================

def analyze_image_with_vision(llm: ChatOpenAI, image_bytes: bytes) -> str:

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    vision_prompt = """
    **ROLE:** Expert Chiromancy (Palmistry) Archivist.

    **TASK:** Analyze this scientific diagram from a palmistry book.

    **INSTRUCTIONS:**
    1. Identify the specific line, mount, or hand shape shown.
    2. Describe length, depth, curvature of lines technically.
    3. Locate Marks (Stars, Crosses, Islands) relative to mounts accurately.
    4. Read any labels (A, B, C, numbers) if present in the diagram.
    5. Note any arrows or directional indicators.

    **OUTPUT FORMAT:** 
    A single detailed paragraph description. 
    Technical facts only - NO interpretations or predictions.
    Describe as if explaining to a blind person.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": vision_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    )

    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"❌ Görsel analiz hatası: {e}")
        return "[GÖRSEL ANALİZ BAŞARISIZ - API HATASI]"


# ============================================
# SAYFA İŞLEME FONKSİYONU
# ============================================

def process_page(
        page: fitz.Page,
        page_number: int,
        doc: fitz.Document,
        llm: ChatOpenAI,
) -> Optional[str]:

    text_content = page.get_text()

    image_list = page.get_images(full=True)
    visual_descriptions: List[str] = []

    for img_index, img in enumerate(image_list):
        xref = img[0]

        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            if len(image_bytes) < MIN_IMAGE_SIZE:
                logger.debug(f"   ⏭️ Küçük görsel atlandı: {len(image_bytes)} bytes")
                continue

            logger.info(f"   🖼️ Sayfa {page_number} - Görsel {img_index + 1} analiz ediliyor...")
            description = analyze_image_with_vision(llm, image_bytes)
            visual_descriptions.append(f"[DIAGRAM {img_index + 1}]: {description}")

        except Exception as e:
            logger.warning(f"   ⚠️ Görsel çıkarma hatası (sayfa {page_number}): {e}")
            continue

    # --- İçerik Birleştirme ---
    # Format: Sayfa metni + Görsel açıklamaları

    combined_content = f"--- PAGE {page_number} START ---\n"
    combined_content += f"{text_content}\n"

    if visual_descriptions:
        combined_content += "\n--- VISUAL CONTENTS ---\n"
        combined_content += "\n".join(visual_descriptions)

    combined_content += f"\n--- PAGE {page_number} END ---\n"

    # Çok kısa içerikleri atla (boş sayfalar vb.)
    if len(combined_content.strip()) < 50:
        return None

    return combined_content


# ============================================
# PDF İŞLEME FONKSİYONU (Ana Fonksiyon)
# ============================================

def process_pdf(pdf_path: str, llm: ChatOpenAI, embeddings: OpenAIEmbeddings) -> int:

    file_name = os.path.basename(pdf_path)

    # Dosya varlık kontrolü
    if not os.path.exists(pdf_path):
        logger.error(f"❌ Dosya bulunamadı: {pdf_path}")
        return 0

    vector_store = get_vector_store(embeddings)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    saved_count = 0

    logger.info(f"📘 KİTAP İŞLENİYOR: '{file_name}' ({total_pages} sayfa)")

    for page_num, page in enumerate(doc):
        real_page_num = page_num + 1

        if real_page_num % LOG_INTERVAL == 0 or real_page_num == 1:
            logger.info(f"   ⏳ İşleniyor: Sayfa {real_page_num}/{total_pages}")

        combined_content = process_page(page, real_page_num, doc, llm)

        if combined_content is None:
            continue

        # -- MongoDB'ye kaydetme --
        try:
            metadata = {
                "source": file_name,
                "page": real_page_num,
                "type": "hybrid_book_page"
            }

            vector_store.add_texts(
                texts=[combined_content],
                metadatas=[metadata]
            )
            saved_count += 1

        except Exception as e:
            logger.error(f"   ❌ Kayıt hatası (sayfa {real_page_num}): {e}")
            continue

    # PDF'i kapat
    doc.close()

    logger.info(f"✅ TAMAMLANDI: '{file_name}' - {saved_count}/{total_pages} sayfa kaydedildi")
    return saved_count


# ============================================
# TOPLU İŞLEME FONKSİYONU (Batch Process)
# ============================================

def batch_process_pdfs(folder_path: str) -> dict:

    results = {
        "total_files": 0,
        "total_pages": 0,
        "processed_files": [],
        "errors": []
    }

    # Klasör varlık kontrolü
    if not os.path.exists(folder_path):
        logger.error(f"❌ Klasör bulunamadı: {folder_path}")
        logger.info(f"💡 Lütfen '{folder_path}' klasörünü oluşturup PDF'leri içine koyun")
        return results

    # PDF dosyalarını bul
    all_files = os.listdir(folder_path)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.warning(f"⚠️ '{folder_path}' klasöründe PDF bulunamadı")
        return results

    results["total_files"] = len(pdf_files)
    logger.info(f"📂 {len(pdf_files)} adet PDF bulundu")

    llm, embeddings = initialize_models()

    # Her PDF'i sırayla işle
    for index, pdf_file in enumerate(pdf_files, start=1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"📖 [{index}/{len(pdf_files)}] İşleniyor: {pdf_file}")
        logger.info(f"{'=' * 50}")

        full_path = os.path.join(folder_path, pdf_file)

        try:
            pages_saved = process_pdf(full_path, llm, embeddings)
            results["total_pages"] += pages_saved
            results["processed_files"].append({
                "file": pdf_file,
                "pages": pages_saved
            })
        except Exception as e:
            error_msg = f"{pdf_file}: {str(e)}"
            logger.error(f"❌ İşlem hatası: {error_msg}")
            results["errors"].append(error_msg)

    return results


# ============================================
# ANA GİRİŞ NOKTASI (Main Entry Point)
# ============================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔮 YASAA VISION - PDF Ingestion Pipeline Başlatılıyor")
    logger.info("=" * 60)

    # 1. Environment doğrulama
    if not validate_environment():
        logger.error("❌ Başlatma başarısız - Environment hataları düzeltilmeli")
        exit(1)

    # 2. PDF klasör yolunu belirle
    # Not: Script App/ingest/ içinde, PDF'ler App/pdf_storage/ içinde
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script dizini
    app_dir = os.path.dirname(script_dir)  # App dizini
    pdf_folder = os.path.join(app_dir, PDF_FOLDER)  # PDF klasör yolu

    logger.info(f"📁 PDF Klasörü: {pdf_folder}")

    # 3. Toplu işleme başlat
    results = batch_process_pdfs(pdf_folder)

    # 4. Sonuç özeti
    logger.info("\n" + "=" * 60)
    logger.info("📊 İŞLEM ÖZETİ")
    logger.info("=" * 60)
    logger.info(f"   📚 Toplam PDF: {results['total_files']}")
    logger.info(f"   📄 Kaydedilen Sayfa: {results['total_pages']}")
    logger.info(f"   ❌ Hata Sayısı: {len(results['errors'])}")

    if results['errors']:
        logger.warning("   ⚠️ Hatalar:")
        for err in results['errors']:
            logger.warning(f"      - {err}")

    logger.info("=" * 60)
    logger.info("🎉 İşlem tamamlandı!")
















