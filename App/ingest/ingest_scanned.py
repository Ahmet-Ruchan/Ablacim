"""
============================================
YASAA VISION - Scanned PDF Ingest Pipeline
============================================
Bu dosya TARANMIŞ (scanned) PDF'leri işler.

Normal PDF'lerden farkı:
- page.get_text() çalışmaz (metin yok, sadece resim var)
- Her sayfa resme çevrilir (render)
- GPT-4o Vision ile metin çıkarılır (OCR + Analiz)
- Sonuç embedding'e çevrilip MongoDB'ye kaydedilir

Kullanım:
    python -m App.ingest.ingest_scanned

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

import os
import sys
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# ENVIRONMENT
# ============================================
load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MONGO_URI: str = os.getenv("MONGO_URI", "")
DB_NAME: str = os.getenv("DB_NAME", "YasaaVisionDB")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "palmistry_knowledge")
INDEX_NAME: str = os.getenv("INDEX_NAME", "vector_index")

# Scanned PDF klasörü (ayrı tutuyoruz)
SCANNED_PDF_FOLDER: str = os.getenv("SCANNED_PDF_FOLDER", "App/pdf_storage/scanned")

# Vision ayarları
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
VISION_MAX_TOKENS: int = int(os.getenv("VISION_MAX_TOKENS", "2000"))

# Render ayarları
RENDER_ZOOM: float = float(os.getenv("RENDER_ZOOM", "2.0"))  # 2x zoom = daha net görüntü

# ============================================
# VISION PROMPT (Taranmış Sayfa İçin)
# ============================================
SCANNED_PAGE_PROMPT = """
You are an expert OCR system specialized in palmistry books.

Analyze this scanned book page and extract ALL content:

1. **TEXT EXTRACTION:**
   - Extract ALL readable text from the page
   - Preserve paragraph structure
   - Include headings, subheadings, and captions
   - Transcribe any handwritten notes if visible

2. **DIAGRAM/ILLUSTRATION ANALYSIS:**
   - If there are hand diagrams, describe them in detail
   - Identify and name any palm lines shown (Heart Line, Head Line, Life Line, Fate Line, etc.)
   - Describe mounts, fingers, and special markings
   - Note any numbered labels or annotations on diagrams

3. **OUTPUT FORMAT:**
   - Write in clear, structured paragraphs
   - Use [DIAGRAM: ...] tags for illustration descriptions
   - Use [FIGURE X: ...] for numbered figures
   - Preserve the logical flow of the original page

IMPORTANT: 
- This is a SCANNED page, so quality may vary
- Extract EVERYTHING you can read
- If text is unclear, make your best interpretation and note [unclear]
- Output should be in the SAME LANGUAGE as the source (Turkish or English)
"""


# ============================================
# HELPER FUNCTIONS
# ============================================
def get_mongo_collection():
    """MongoDB koleksiyonuna bağlanır."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def get_vector_store() -> MongoDBAtlasVectorSearch:
    """MongoDB Vector Store'u döndürür."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key="text",
        embedding_key="embedding"
    )

    return vector_store


def render_page_to_image(page: fitz.Page, zoom: float = 2.0) -> bytes:
    """
    PDF sayfasını PNG resme çevirir.

    Args:
        page: PyMuPDF sayfa objesi
        zoom: Büyütme faktörü (2.0 = 2x çözünürlük)

    Returns:
        bytes: PNG formatında resim
    """
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")


def analyze_page_with_vision(llm: ChatOpenAI, image_bytes: bytes) -> str:
    """
    Taranmış sayfa resmini GPT-4o Vision ile analiz eder.

    Args:
        llm: ChatOpenAI instance
        image_bytes: PNG formatında sayfa resmi

    Returns:
        str: Sayfadan çıkarılan metin ve açıklamalar
    """
    # Base64'e çevir
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Vision API çağrısı
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SCANNED_PAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"  # Yüksek detay modu
                    }
                }
            ]
        }
    ]

    response = llm.invoke(messages)
    return response.content


def process_scanned_pdf(
        pdf_path: Path,
        llm: ChatOpenAI,
        vector_store: MongoDBAtlasVectorSearch
) -> Dict[str, Any]:
    """
    Tek bir taranmış PDF'i işler.

    Args:
        pdf_path: PDF dosya yolu
        llm: ChatOpenAI instance
        vector_store: MongoDB Vector Store

    Returns:
        Dict: İşlem istatistikleri
    """
    stats = {
        "file_name": pdf_path.name,
        "total_pages": 0,
        "processed_pages": 0,
        "failed_pages": 0,
        "documents_added": 0
    }

    logger.info(f"📖 PDF açılıyor: {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
        stats["total_pages"] = len(doc)
        logger.info(f"   📄 Toplam sayfa: {len(doc)}")

        documents_to_add: List[Document] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            real_page_num = page_num + 1

            logger.info(f"   🔄 Sayfa {real_page_num}/{len(doc)} işleniyor...")

            try:
                # 1. Sayfayı resme çevir
                image_bytes = render_page_to_image(page, zoom=RENDER_ZOOM)
                logger.info(f"      📸 Sayfa render edildi ({len(image_bytes)} bytes)")

                # 2. GPT-4o Vision ile analiz et
                extracted_text = analyze_page_with_vision(llm, image_bytes)
                logger.info(f"      🔍 Vision analizi tamamlandı ({len(extracted_text)} karakter)")

                # 3. Boş kontrolü
                if len(extracted_text.strip()) < 50:
                    logger.warning(f"      ⚠️ Sayfa {real_page_num} çok az içerik, atlanıyor...")
                    continue

                # 4. Document oluştur
                metadata = {
                    "source": pdf_path.name,
                    "page": real_page_num,
                    "type": "scanned_book_page",
                    "processed_at": datetime.now().isoformat(),
                    "vision_model": VISION_MODEL
                }

                document = Document(
                    page_content=extracted_text,
                    metadata=metadata
                )

                documents_to_add.append(document)
                stats["processed_pages"] += 1

                logger.info(f"      ✅ Sayfa {real_page_num} başarıyla işlendi")

            except Exception as e:
                logger.error(f"      ❌ Sayfa {real_page_num} hatası: {e}")
                stats["failed_pages"] += 1
                continue

        # 5. MongoDB'ye toplu kaydet
        if documents_to_add:
            logger.info(f"   💾 {len(documents_to_add)} döküman MongoDB'ye kaydediliyor...")
            vector_store.add_documents(documents_to_add)
            stats["documents_added"] = len(documents_to_add)
            logger.info(f"   ✅ Kayıt tamamlandı!")

        doc.close()

    except Exception as e:
        logger.error(f"❌ PDF işleme hatası: {e}")
        raise

    return stats


def find_scanned_pdfs(folder_path: str) -> List[Path]:
    """Klasördeki PDF dosyalarını bulur."""
    folder = Path(folder_path)

    if not folder.exists():
        logger.warning(f"⚠️ Klasör bulunamadı, oluşturuluyor: {folder_path}")
        folder.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = list(folder.glob("*.pdf"))
    return pdf_files


# ============================================
# MAIN
# ============================================
def main():
    """Ana çalıştırma fonksiyonu."""

    logger.info("=" * 60)
    logger.info("🔮 YASAA VISION - Scanned PDF Ingest Pipeline")
    logger.info("=" * 60)

    # Kontroller
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY bulunamadı!")
        sys.exit(1)

    if not MONGO_URI:
        logger.error("❌ MONGO_URI bulunamadı!")
        sys.exit(1)

    # PDF'leri bul
    pdf_files = find_scanned_pdfs(SCANNED_PDF_FOLDER)

    if not pdf_files:
        logger.warning(f"⚠️ {SCANNED_PDF_FOLDER} klasöründe PDF bulunamadı!")
        logger.info(f"   Taranmış PDF'lerinizi şu klasöre koyun: {SCANNED_PDF_FOLDER}")
        sys.exit(0)

    logger.info(f"📚 {len(pdf_files)} adet PDF bulundu")

    # LLM ve Vector Store oluştur
    llm = ChatOpenAI(
        model=VISION_MODEL,
        max_tokens=VISION_MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = get_vector_store()

    # Her PDF'i işle
    all_stats = []

    for pdf_path in pdf_files:
        logger.info("-" * 40)
        try:
            stats = process_scanned_pdf(pdf_path, llm, vector_store)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"❌ {pdf_path.name} işlenemedi: {e}")
            all_stats.append({
                "file_name": pdf_path.name,
                "error": str(e)
            })

    # Özet
    logger.info("=" * 60)
    logger.info("📊 İŞLEM ÖZETİ")
    logger.info("=" * 60)

    total_pages = 0
    total_processed = 0
    total_failed = 0
    total_docs = 0

    for stat in all_stats:
        if "error" in stat:
            logger.error(f"   ❌ {stat['file_name']}: HATA - {stat['error']}")
        else:
            logger.info(f"   ✅ {stat['file_name']}:")
            logger.info(
                f"      Sayfa: {stat['total_pages']} | İşlenen: {stat['processed_pages']} | Hata: {stat['failed_pages']}")
            total_pages += stat["total_pages"]
            total_processed += stat["processed_pages"]
            total_failed += stat["failed_pages"]
            total_docs += stat["documents_added"]

    logger.info("-" * 40)
    logger.info(
        f"📈 TOPLAM: {total_pages} sayfa | {total_processed} işlendi | {total_failed} hata | {total_docs} döküman")
    logger.info("=" * 60)
    logger.info("✅ Scanned PDF Ingest tamamlandı!")


if __name__ == "__main__":
    main()
