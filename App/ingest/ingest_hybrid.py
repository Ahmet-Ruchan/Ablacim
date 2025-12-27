"""
============================================
YASAA VISION - Hybrid PDF Ingest Pipeline
============================================
Bu dosya hem text-based hem taranmış PDF'leri akıllıca işler.

Özellikler:
- Text varsa → Text'i al
- Text yoksa → Vision ile oku
- Gömülü resim varsa → HER ZAMAN Vision'a gönder
- Diyagram keyword varsa → Sayfayı render edip Vision'a gönder
- Overlap → Sayfa geçişlerinde bağlam korunur

Kullanım:
    python -m App.ingest.ingest_hybrid

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

import os
import sys
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
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

# PDF klasörü
PDF_FOLDER: str = os.getenv("PDF_FOLDER", "pdf_storage")

# Model ayarları
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
VISION_MAX_TOKENS: int = int(os.getenv("VISION_MAX_TOKENS", "3000"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Hibrit ayarlar
MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "500"))
RENDER_ZOOM: float = float(os.getenv("RENDER_ZOOM", "2.0"))
OVERLAP_SIZE: int = int(os.getenv("OVERLAP_SIZE", "500"))
MIN_IMAGE_SIZE: int = int(os.getenv("MIN_IMAGE_SIZE", "3000"))

# Diyagram anahtar kelimeleri
DIAGRAM_KEYWORDS = [
    # İngilizce
    "Fig", "Figure", "Plate", "Diagram", "See illustration",
    "Mount", "Line", "drawing", "sketch", "chart", "image",
    # Türkçe
    "Şekil", "Çizim", "Resim", "Diyagram", "Tablo",
    "Çizgi", "Tepe", "şekilde", "görüldüğü", "bakınız"
]


# ============================================
# VISION PROMPTS
# ============================================
VISION_PROMPT_FULL_PAGE = """
You are an expert OCR and content extraction system for palmistry books.

This page appears to have little or no extractable text (it may be a scanned image).
Please extract ALL content from this page:

1. **TEXT:** Transcribe all readable text, preserving structure
2. **DIAGRAMS:** Describe any hand diagrams, palm lines, or illustrations in detail
3. **LABELS:** Note any numbered figures, labels, or annotations

Output in the SAME LANGUAGE as the source content.
If text is unclear, write [unclear] but make your best interpretation.
"""

VISION_PROMPT_DIAGRAM_ONLY = """
You are an expert at analyzing palmistry diagrams and illustrations.

This page has text that was already extracted. Now focus ONLY on the visual elements:

1. **DIAGRAMS:** Describe any hand drawings, palm diagrams, or illustrations
2. **LINES:** Identify specific palm lines shown (Heart, Head, Life, Fate, etc.)
3. **MOUNTS:** Describe any mounts or areas highlighted
4. **LABELS:** Note figure numbers and what they show

Be technical and detailed. Output in the same language as the page.
If no diagrams exist, respond with: "NO_DIAGRAMS_FOUND"
"""

VISION_PROMPT_EMBEDDED_IMAGE = """
Analyze this image from a palmistry book.

Describe in detail:
1. What type of image is this? (hand diagram, palm lines, mount illustration, etc.)
2. What specific features are shown? (lines, mounts, fingers, markings)
3. Any labels, numbers, or text visible in the image
4. The educational purpose of this illustration

Be technical and precise. Output in the same language as typical palmistry terminology.
"""


# ============================================
# HELPER FUNCTIONS
# ============================================
def get_vector_store() -> MongoDBAtlasVectorSearch:
    """MongoDB Vector Store'u döndürür."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key="text",
        embedding_key="embedding"
    )


def render_page_to_image(page: fitz.Page, zoom: float = 2.0) -> bytes:
    """PDF sayfasını PNG resme çevirir."""
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")


def analyze_with_vision(llm: ChatOpenAI, image_bytes: bytes, prompt: str) -> str:
    """Resmi GPT-4o Vision ile analiz eder."""
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = llm.invoke(messages)
    return response.content


def has_diagram_keywords(text: str) -> bool:
    """Metinde diyagram anahtar kelimeleri var mı kontrol eder."""
    text_lower = text.lower()
    for keyword in DIAGRAM_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def extract_embedded_images(
    page: fitz.Page,
    doc: fitz.Document,
    llm: ChatOpenAI,
    page_num: int
) -> List[str]:
    """
    Sayfadaki gömülü resimleri bulur ve Vision ile analiz eder.

    HER ZAMAN ÇALIŞIR - Resim varsa Vision'a gönderir!

    Args:
        page: PyMuPDF sayfa objesi
        doc: PyMuPDF döküman objesi
        llm: ChatOpenAI instance
        page_num: Sayfa numarası

    Returns:
        List[str]: Her resim için Vision açıklamaları
    """
    descriptions = []

    try:
        images = page.get_images(full=True)

        if not images:
            return descriptions

        logger.info(f"      🖼️ {len(images)} gömülü resim bulundu")

        for img_idx, img_info in enumerate(images):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Çok küçük resimleri atla (ikonlar, süslemeler)
                if  50 < len(image_bytes) < MIN_IMAGE_SIZE:
                    logger.info(f"         ⏭️ Resim {img_idx+1} çok küçük ({len(image_bytes)} bytes), atlanıyor")
                    continue

                logger.info(f"         🔍 Resim {img_idx+1} Vision'a gönderiliyor ({len(image_bytes)} bytes)")

                # Vision'a gönder
                description = analyze_with_vision(llm, image_bytes, VISION_PROMPT_EMBEDDED_IMAGE)
                descriptions.append(f"[IMAGE {img_idx+1} - Page {page_num}]: {description}")

                logger.info(f"         ✅ Resim {img_idx+1} analiz edildi")

            except Exception as e:
                logger.warning(f"         ⚠️ Resim {img_idx+1} hatası: {e}")
                continue

    except Exception as e:
        logger.warning(f"      ⚠️ Resim çıkarma hatası: {e}")

    return descriptions


def process_page_hybrid(
    page: fitz.Page,
    doc: fitz.Document,
    page_num: int,
    llm: ChatOpenAI,
    previous_tail: str = ""
) -> Tuple[str, str, str]:
    """
    Sayfayı hibrit şekilde işler.

    Mantık:
    1. Her zaman text çıkarmayı dene
    2. Her zaman gömülü resimleri ara ve Vision'a gönder
    3. Text azsa veya diyagram keyword varsa sayfayı da render et
    4. Hepsini birleştir + Overlap ekle

    Args:
        page: PyMuPDF sayfa objesi
        doc: PyMuPDF döküman objesi
        page_num: Sayfa numarası
        llm: ChatOpenAI instance
        previous_tail: Önceki sayfanın son 500 karakteri (overlap için)

    Returns:
        Tuple[content, new_tail, mode]: İşlenmiş içerik, yeni overlap, işlem modu
    """

    # ========== 1. TEXT ÇIKAR ==========
    raw_text = page.get_text().strip()
    text_length = len(raw_text)
    logger.info(f"      📝 Text: {text_length} karakter")

    # ========== 2. GÖMÜLÜ RESİMLERİ ANALİZ ET (HER ZAMAN) ==========
    image_descriptions = extract_embedded_images(page, doc, llm, page_num)
    has_images = len(image_descriptions) > 0

    # ========== 3. EK ANALİZ GEREKİYOR MU? ==========
    page_render_description = ""
    processing_mode = "TEXT_ONLY"

    # Durum A: Text çok az → Sayfayı komple render et
    if text_length < MIN_TEXT_LENGTH:
        processing_mode = "VISION_FULL"
        logger.info(f"      🔍 Mode: VISION_FULL (text yetersiz, sayfa render ediliyor)")

        image_bytes = render_page_to_image(page, RENDER_ZOOM)
        page_render_description = analyze_with_vision(llm, image_bytes, VISION_PROMPT_FULL_PAGE)

    # Durum B: Text var ama diyagram keyword var → Sayfayı da render et
    elif has_diagram_keywords(raw_text):
        processing_mode = "HYBRID"
        logger.info(f"      🔍 Mode: HYBRID (text + diyagram keyword bulundu)")

        image_bytes = render_page_to_image(page, RENDER_ZOOM)
        vision_result = analyze_with_vision(llm, image_bytes, VISION_PROMPT_DIAGRAM_ONLY)

        if "NO_DIAGRAMS_FOUND" not in vision_result:
            page_render_description = vision_result

    # Durum C: Text var + resim var
    elif has_images:
        processing_mode = "TEXT_WITH_IMAGES"
        logger.info(f"      🔍 Mode: TEXT_WITH_IMAGES (text + gömülü resimler)")

    # Durum D: Sadece text
    else:
        processing_mode = "TEXT_ONLY"
        logger.info(f"      🔍 Mode: TEXT_ONLY")

    # ========== 4. HEPSİNİ BİRLEŞTİR ==========
    final_content = ""

    # 4a. Overlap (önceki sayfadan gelen kuyruk)
    if previous_tail and OVERLAP_SIZE > 0:
        final_content += f"[...önceki sayfadan devam...]\n{previous_tail}\n\n"

    # 4b. Ana text (veya vision full page sonucu)
    if processing_mode == "VISION_FULL":
        # Text yok, vision sonucunu ana içerik olarak kullan
        final_content += page_render_description
    else:
        # Text var, onu ana içerik olarak kullan
        if raw_text:
            final_content += raw_text

        # Sayfa render açıklaması (diyagram analizi)
        if page_render_description:
            final_content += f"\n\n[DIAGRAM ANALYSIS]\n{page_render_description}"

    # 4c. Gömülü resim açıklamaları
    if image_descriptions:
        final_content += f"\n\n[EMBEDDED IMAGES]\n"
        final_content += "\n\n".join(image_descriptions)

    # ========== 5. OVERLAP HAZIRLA (Sonraki sayfa için) ==========
    # Not: Overlap için sadece raw_text kullanılır (vision açıklamaları değil)
    if len(raw_text) > OVERLAP_SIZE:
        new_tail = raw_text[-OVERLAP_SIZE:]
    else:
        new_tail = raw_text

    return final_content, new_tail, processing_mode


def process_pdf(
    pdf_path: Path,
    llm: ChatOpenAI,
    vector_store: MongoDBAtlasVectorSearch
) -> Dict[str, Any]:
    """Tek bir PDF'i işler."""

    stats = {
        "file_name": pdf_path.name,
        "total_pages": 0,
        "text_only_pages": 0,
        "text_with_images_pages": 0,
        "vision_full_pages": 0,
        "hybrid_pages": 0,
        "skipped_pages": 0,
        "documents_added": 0,
        "total_images_analyzed": 0
    }

    logger.info(f"📖 PDF açılıyor: {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
        stats["total_pages"] = len(doc)
        logger.info(f"   📄 Toplam sayfa: {len(doc)}")

        documents: List[Document] = []
        previous_tail = ""  # Overlap için

        for page_num in range(len(doc)):
            page = doc[page_num]
            real_page = page_num + 1

            logger.info(f"   🔄 Sayfa {real_page}/{len(doc)} işleniyor...")

            try:
                # Hibrit işleme
                content, previous_tail, mode = process_page_hybrid(
                    page=page,
                    doc=doc,
                    page_num=real_page,
                    llm=llm,
                    previous_tail=previous_tail
                )

                # İstatistik güncelle
                if mode == "TEXT_ONLY":
                    stats["text_only_pages"] += 1
                elif mode == "TEXT_WITH_IMAGES":
                    stats["text_with_images_pages"] += 1
                elif mode == "VISION_FULL":
                    stats["vision_full_pages"] += 1
                elif mode == "HYBRID":
                    stats["hybrid_pages"] += 1

                # Resim sayısını say
                if "[IMAGE" in content:
                    image_count = content.count("[IMAGE")
                    stats["total_images_analyzed"] += image_count

                # Boş kontrolü
                if len(content.strip()) < 50:
                    logger.warning(f"      ⚠️ İçerik çok kısa, atlanıyor")
                    stats["skipped_pages"] += 1
                    continue

                # Document oluştur
                metadata = {
                    "source": pdf_path.name,
                    "page": real_page,
                    "type": "hybrid_book_page",
                    "processing_mode": mode,
                    "has_overlap": OVERLAP_SIZE > 0,
                    "processed_at": datetime.now().isoformat()
                }

                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

                logger.info(f"      ✅ Sayfa {real_page} tamamlandı [{mode}]")

            except Exception as e:
                logger.error(f"      ❌ Sayfa {real_page} hatası: {e}")
                stats["skipped_pages"] += 1
                continue

        # MongoDB'ye kaydet
        if documents:
            logger.info(f"   💾 {len(documents)} döküman MongoDB'ye kaydediliyor...")
            vector_store.add_documents(documents)
            stats["documents_added"] = len(documents)
            logger.info(f"   ✅ Kayıt tamamlandı!")

        doc.close()

    except Exception as e:
        logger.error(f"❌ PDF hatası: {e}")
        raise

    return stats


def find_pdfs(folder: str) -> List[Path]:
    """Klasördeki PDF'leri bulur (alt klasörler dahil)."""
    path = Path(folder)
    if not path.exists():
        logger.warning(f"⚠️ Klasör bulunamadı, oluşturuluyor: {folder}")
        path.mkdir(parents=True, exist_ok=True)
        return []

    # Hem ana klasör hem alt klasörlerdeki PDF'ler
    return list(path.rglob("*.pdf"))


# ============================================
# MAIN
# ============================================
def main():
    logger.info("=" * 60)
    logger.info("🔮 YASAA VISION - Hybrid PDF Ingest Pipeline")
    logger.info("=" * 60)
    logger.info(f"📁 PDF Klasörü: {PDF_FOLDER}")
    logger.info(f"🔧 Ayarlar:")
    logger.info(f"   - Min Text Length: {MIN_TEXT_LENGTH} karakter")
    logger.info(f"   - Overlap Size: {OVERLAP_SIZE} karakter")
    logger.info(f"   - Min Image Size: {MIN_IMAGE_SIZE} bytes")
    logger.info(f"   - Render Zoom: {RENDER_ZOOM}x")
    logger.info("=" * 60)

    # Kontroller
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY bulunamadı!")
        sys.exit(1)

    if not MONGO_URI:
        logger.error("❌ MONGO_URI bulunamadı!")
        sys.exit(1)

    logger.info("✅ API anahtarları mevcut")

    # PDF'leri bul
    pdf_files = find_pdfs(PDF_FOLDER)

    if not pdf_files:
        logger.warning(f"⚠️ {PDF_FOLDER} klasöründe PDF bulunamadı!")
        logger.info(f"   PDF dosyalarınızı şu klasöre koyun: {PDF_FOLDER}")
        sys.exit(0)

    logger.info(f"📚 {len(pdf_files)} adet PDF bulundu:")
    for pdf in pdf_files:
        logger.info(f"   📄 {pdf.name}")

    # LLM ve Vector Store
    logger.info("-" * 40)
    logger.info("🤖 Modeller yükleniyor...")

    llm = ChatOpenAI(
        model=VISION_MODEL,
        max_tokens=VISION_MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY
    )
    logger.info(f"   ✅ Vision Model: {VISION_MODEL}")

    vector_store = get_vector_store()
    logger.info(f"   ✅ MongoDB Vector Store: {DB_NAME}/{COLLECTION_NAME}")

    # Her PDF'i işle
    all_stats = []

    for pdf_path in pdf_files:
        logger.info("=" * 60)
        try:
            stats = process_pdf(pdf_path, llm, vector_store)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"❌ {pdf_path.name} işlenemedi: {e}")
            all_stats.append({"file_name": pdf_path.name, "error": str(e)})

    # ========== ÖZET ==========
    logger.info("=" * 60)
    logger.info("📊 İŞLEM ÖZETİ")
    logger.info("=" * 60)

    total_pages = 0
    total_text_only = 0
    total_text_with_images = 0
    total_vision_full = 0
    total_hybrid = 0
    total_skipped = 0
    total_docs = 0
    total_images = 0

    for s in all_stats:
        if "error" in s:
            logger.error(f"   ❌ {s['file_name']}: HATA - {s['error']}")
        else:
            logger.info(f"   ✅ {s['file_name']}:")
            logger.info(f"      Sayfa: {s['total_pages']}")
            logger.info(f"      Text-Only: {s['text_only_pages']} | Text+Images: {s['text_with_images_pages']}")
            logger.info(f"      Vision-Full: {s['vision_full_pages']} | Hybrid: {s['hybrid_pages']}")
            logger.info(f"      Resim Analizi: {s['total_images_analyzed']} | Atlanan: {s['skipped_pages']}")

            total_pages += s["total_pages"]
            total_text_only += s["text_only_pages"]
            total_text_with_images += s["text_with_images_pages"]
            total_vision_full += s["vision_full_pages"]
            total_hybrid += s["hybrid_pages"]
            total_skipped += s["skipped_pages"]
            total_docs += s["documents_added"]
            total_images += s["total_images_analyzed"]

    logger.info("-" * 40)
    logger.info(f"📈 GENEL TOPLAM:")
    logger.info(f"   📄 Toplam Sayfa: {total_pages}")
    logger.info(f"   📝 Text-Only: {total_text_only}")
    logger.info(f"   🖼️ Text+Images: {total_text_with_images}")
    logger.info(f"   👁️ Vision-Full: {total_vision_full}")
    logger.info(f"   🔀 Hybrid: {total_hybrid}")
    logger.info(f"   ⏭️ Atlanan: {total_skipped}")
    logger.info(f"   🖼️ Analiz Edilen Resim: {total_images}")
    logger.info(f"   💾 MongoDB'ye Kaydedilen: {total_docs} döküman")
    logger.info("=" * 60)
    logger.info("✅ Hybrid Ingest tamamlandı!")


if __name__ == "__main__":
    main()
