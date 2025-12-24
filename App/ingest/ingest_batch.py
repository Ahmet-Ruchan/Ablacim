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

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os                                      # İşletim sistemi işlemleri (dosya yolları vb.)
import logging                                 # Log yönetimi (print yerine profesyonel loglama)
import base64                                  # Görselleri base64 formatına çevirmek için
from typing import Optional, List              # Type hints için tip tanımlamaları

import fitz                                    # PyMuPDF - PDF işleme kütüphanesi
from dotenv import load_dotenv                 # .env dosyasından değişken okuma
from pymongo import MongoClient                # MongoDB bağlantısı için driver
from langchain_openai import (                 # OpenAI entegrasyonları
    ChatOpenAI,                                # GPT-4o chat modeli
    OpenAIEmbeddings                           # text-embedding-3-small
)
from langchain_core.messages import HumanMessage  # LangChain mesaj formatı
from langchain_mongodb import MongoDBAtlasVectorSearch  # MongoDB vektör arama


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
MIN_IMAGE_SIZE: int = int(os.getenv("MIN_IMAGE_SIZE", "5000"))          # Min görsel boyutu (byte)
LOG_INTERVAL: int = int(os.getenv("LOG_INTERVAL", "10"))                # Kaç sayfada bir log basılsın

# --- CHUNK OVERLAP AYARLARI ---
OVERLAP_SIZE: int = int(os.getenv("OVERLAP_SIZE", "500"))               # Önceki sayfadan kaç karakter alınacak
"""
OVERLAP_SIZE: Sayfa sınırlarında bağlam kopukluğunu önler.
- 500: ~3-4 cümle (önerilen)
- 1000: ~6-8 cümle (daha fazla bağlam ama daha fazla token)
- 0: Overlap kapalı

Nasıl çalışır:
    Sayfa 49'un son 500 karakteri → Sayfa 50'nin başına eklenir
    Böylece cümle ortasında kopma sorunu çözülür.
"""


# ============================================
# DOĞRULAMA - Kritik değişkenler var mı?
# ============================================
def validate_environment() -> bool:
    """
    Kritik environment değişkenlerinin varlığını kontrol eder.

    Returns:
        bool: Tüm değişkenler mevcutsa True, değilse False
    """
    # Kontrol edilecek kritik değişkenler
    required_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MONGO_URI": MONGO_URI
    }

    missing = []  # Eksik değişkenleri topla

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
    """
    OpenAI modellerini başlatır.

    Returns:
        tuple: (ChatOpenAI instance, OpenAIEmbeddings instance)
    """
    logger.info(f"🤖 Modeller yükleniyor: Vision={VISION_MODEL}, Embedding={EMBEDDING_MODEL}")

    # GPT-4o Vision modeli (görsel analiz için)
    llm = ChatOpenAI(
        model=VISION_MODEL,           # gpt-4o
        api_key=OPENAI_API_KEY,       # API anahtarı
        max_tokens=MAX_TOKENS         # Maksimum çıktı token sayısı
    )

    # Embedding modeli (vektörleştirme için)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,        # text-embedding-3-small
        api_key=OPENAI_API_KEY        # API anahtarı
    )

    logger.info("✅ Modeller başarıyla yüklendi")
    return llm, embeddings


# ============================================
# MONGODB VECTOR STORE BAĞLANTISI
# ============================================
def get_vector_store(embeddings: OpenAIEmbeddings) -> MongoDBAtlasVectorSearch:
    """
    MongoDB Atlas Vector Store bağlantısını oluşturur.

    Args:
        embeddings: OpenAI embedding modeli instance'ı

    Returns:
        MongoDBAtlasVectorSearch: Vektör store instance'ı
    """
    logger.info(f"🔌 MongoDB'ye bağlanılıyor: {DB_NAME}/{COLLECTION_NAME}")

    # MongoDB client oluştur
    client = MongoClient(MONGO_URI)

    # Collection referansını al
    collection = client[DB_NAME][COLLECTION_NAME]

    # Vector store oluştur
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,        # MongoDB collection
        embedding=embeddings,         # Embedding modeli
        index_name=INDEX_NAME         # Atlas Search index adı
    )

    logger.info("✅ MongoDB bağlantısı kuruldu")
    return vector_store


# ============================================
# GÖRSEL ANALİZ FONKSİYONU (GPT-4o Vision)
# ============================================
def analyze_image_with_vision(
    llm: ChatOpenAI,
    image_bytes: bytes
) -> str:
    """
    Bir görseli GPT-4o Vision modeli ile analiz eder.
    El falı diyagramlarını teknik olarak açıklar.

    Args:
        llm: ChatOpenAI instance (GPT-4o)
        image_bytes: Görselin binary verisi

    Returns:
        str: Görselin teknik açıklaması
    """
    # Görseli base64 formatına çevir (API için gerekli)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Vision analiz prompt'u
    # NOT: Yorum değil, sadece teknik betimleme istenmiş
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

    # LangChain mesaj formatında hazırla
    message = HumanMessage(
        content=[
            {"type": "text", "text": vision_prompt},  # Metin talimatı
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"  # Base64 görsel
                }
            },
        ]
    )

    # API çağrısı - hata yakalama ile
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"❌ Görsel analiz hatası: {e}")
        return "[GÖRSEL ANALİZ BAŞARISIZ - API Hatası]"


# ============================================
# SAYFA İŞLEME FONKSİYONU
# ============================================
def process_page(
    page: fitz.Page,
    page_number: int,
    doc: fitz.Document,
    llm: ChatOpenAI
) -> Optional[str]:
    """
    Tek bir PDF sayfasını işler: metin + görseller.

    Args:
        page: PyMuPDF sayfa objesi
        page_number: Sayfa numarası (1'den başlar)
        doc: PDF doküman objesi (görsel çıkarmak için)
        llm: ChatOpenAI instance

    Returns:
        Optional[str]: Birleştirilmiş içerik veya None
    """
    # --- Metin Çıkarma ---
    text_content = page.get_text()

    # --- Görsel Çıkarma ve Analiz ---
    image_list = page.get_images(full=True)  # Sayfadaki tüm görseller
    visual_descriptions: List[str] = []       # Görsel açıklamaları toplayacak liste

    # Her görseli işle
    for img_index, img in enumerate(image_list):
        xref = img[0]  # Görsel referans ID'si

        try:
            # Görseli çıkar
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # --- FİLTRELEME: Küçük görselleri atla (logo, ikon vb.) ---
            if len(image_bytes) < MIN_IMAGE_SIZE:
                logger.debug(f"   ⏭️ Küçük görsel atlandı: {len(image_bytes)} bytes")
                continue

            # GPT-4o ile analiz et
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

    # Görsel açıklamaları varsa ekle
    if visual_descriptions:
        combined_content += "\n--- VISUAL CONTENTS ---\n"
        combined_content += "\n".join(visual_descriptions)

    combined_content += f"\n--- PAGE {page_number} END ---"

    # Çok kısa içerikleri atla (boş sayfalar vb.)
    if len(combined_content.strip()) < 50:
        return None

    return combined_content


# ============================================
# PDF İŞLEME FONKSİYONU (OVERLAP DESTEKLİ)
# ============================================
def process_pdf(
    pdf_path: str,
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings
) -> int:
    """
    Tek bir PDF dosyasını OVERLAP (örtüşme) desteğiyle işler.

    OVERLAP NEDİR?
    Sayfa 49'un sonu: "...akıl çizgisi çatallı ise bu kişi..."
    Sayfa 50'nin başı: "[Önceki sayfadan:] ...çatallı ise bu kişi..." + "...yaratıcı düşünce..."

    Bu sayede:
    - Cümle ortasında kopma sorunu çözülür
    - Embedding modeli bağlamı anlar
    - Arama kalitesi artar

    Args:
        pdf_path: PDF dosyasının tam yolu
        llm: ChatOpenAI instance
        embeddings: OpenAIEmbeddings instance

    Returns:
        int: Başarıyla kaydedilen sayfa sayısı
    """
    file_name = os.path.basename(pdf_path)

    # Dosya varlık kontrolü
    if not os.path.exists(pdf_path):
        logger.error(f"❌ Dosya bulunamadı: {pdf_path}")
        return 0

    # Vector store bağlantısı al
    vector_store = get_vector_store(embeddings)

    # PDF'i aç
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    saved_count = 0

    logger.info(f"📘 KİTAP İŞLENİYOR: '{file_name}' ({total_pages} sayfa)")
    logger.info(f"   🔗 Overlap aktif: {OVERLAP_SIZE} karakter")

    # ==========================================
    # OVERLAP İÇİN HAFIZA DEĞİŞKENİ
    # ==========================================
    # Bir önceki sayfanın son kısmını burada tutuyoruz
    previous_page_text_tail: str = ""

    # Her sayfayı işle
    for page_num, page in enumerate(doc):
        real_page_num = page_num + 1

        # İlerleme logu
        if real_page_num % LOG_INTERVAL == 0 or real_page_num == 1:
            logger.info(f"   ⏳ İşleniyor: Sayfa {real_page_num}/{total_pages}")

        # ==========================================
        # ADIM 1: Mevcut Sayfanın Metnini Al
        # ==========================================
        current_page_text = page.get_text()

        # ==========================================
        # ADIM 2: OVERLAP BİRLEŞTİRME
        # ==========================================
        # Embedding için kullanılacak metin:
        # [Önceki Sayfanın Sonu] + [Şu Anki Sayfanın Tamamı]

        text_for_embedding = ""

        # Önceki sayfadan bağlam varsa ekle
        if previous_page_text_tail and OVERLAP_SIZE > 0:
            text_for_embedding += f"\n[...Sayfa {real_page_num - 1}'den devam...]\n"
            text_for_embedding += previous_page_text_tail + "\n"
            text_for_embedding += "[...Sayfa sonu...]\n\n"

        # Şu anki sayfanın metnini ekle
        text_for_embedding += current_page_text

        # ==========================================
        # ADIM 3: BU SAYFANIN SONUNU HAFIZAYA AL
        # ==========================================
        # Bir sonraki sayfa için bu sayfanın sonunu sakla
        if OVERLAP_SIZE > 0:
            if len(current_page_text) > OVERLAP_SIZE:
                # Sayfanın son OVERLAP_SIZE karakterini al
                previous_page_text_tail = current_page_text[-OVERLAP_SIZE:]
            else:
                # Sayfa kısaysa tamamını al
                previous_page_text_tail = current_page_text

        # ==========================================
        # ADIM 4: GÖRSELLERİ İŞLE
        # ==========================================
        image_list = page.get_images(full=True)
        visual_descriptions: List[str] = []

        for img_index, img in enumerate(image_list):
            xref = img[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Küçük görselleri atla
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue

                logger.info(f"   🖼️ Sayfa {real_page_num} - Görsel {img_index + 1} analiz ediliyor...")
                description = analyze_image_with_vision(llm, image_bytes)
                visual_descriptions.append(f"[DIAGRAM {img_index + 1}]: {description}")

            except Exception as e:
                logger.warning(f"   ⚠️ Görsel hatası: {e}")
                continue

        # ==========================================
        # ADIM 5: NİHAİ İÇERİK BİRLEŞTİRME
        # ==========================================
        combined_content = f"--- PAGE {real_page_num} START ---\n"
        combined_content += text_for_embedding  # Artık overlap içeriyor!

        if visual_descriptions:
            combined_content += "\n\n--- VISUAL CONTENTS ---\n"
            combined_content += "\n".join(visual_descriptions)

        combined_content += f"\n--- PAGE {real_page_num} END ---"

        # Boş sayfa kontrolü
        if len(combined_content.strip()) < 50:
            continue

        # ==========================================
        # ADIM 6: MongoDB'ye Kaydet
        # ==========================================
        try:
            metadata = {
                "source": file_name,
                "page": real_page_num,
                "type": "hybrid_book_page",
                "has_overlap": OVERLAP_SIZE > 0  # Overlap bilgisi
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

    logger.info(f"✅ TAMAMLANDI: '{file_name}' - {saved_count}/{total_pages} sayfa (overlap: {OVERLAP_SIZE})")
    return saved_count


# ============================================
# TOPLU İŞLEME FONKSİYONU (Batch Process)
# ============================================
def batch_process_pdfs(folder_path: str) -> dict:
    """
    Bir klasördeki tüm PDF dosyalarını sırayla işler.

    Args:
        folder_path: PDF klasörünün yolu

    Returns:
        dict: İşlem özeti {"total_files": X, "total_pages": Y, "errors": [...]}
    """
    # Sonuç istatistikleri
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

    # Modelleri başlat (bir kez)
    llm, embeddings = initialize_models()

    # Her PDF'i sırayla işle
    for index, pdf_file in enumerate(pdf_files, start=1):
        logger.info(f"\n{'='*50}")
        logger.info(f"📖 [{index}/{len(pdf_files)}] İşleniyor: {pdf_file}")
        logger.info(f"{'='*50}")

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
    app_dir = os.path.dirname(script_dir)                     # App dizini
    pdf_folder = os.path.join(app_dir, PDF_FOLDER)            # PDF klasör yolu

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