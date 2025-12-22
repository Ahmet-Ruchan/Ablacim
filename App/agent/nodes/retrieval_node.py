"""
============================================
YASAA VISION - Retrieval Node (Araştırmacı)
============================================
Bu düğüm, Gözcü'nün teknik raporunu alır ve
MongoDB Atlas'ta semantik arama yaparak ilgili
kitap sayfalarını bulur.

Görev:
- Gözcü'nün raporunu sorgu olarak kullan
- MongoDB Vector Search ile en alakalı sayfaları bul
- Bulunan bilgileri state'e ekle

Çıktı:
- retrieved_documents: Kitaplardan bulunan ilgili sayfalar

Akış:
    Gözcü Raporu → Embedding → Similarity Search → Sonuçlar
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os                                      # Environment değişkenleri için
import logging                                 # Profesyonel loglama
from typing import Dict, Any, List, Optional   # Type hints için

from dotenv import load_dotenv                 # .env dosyası okuma
from pymongo import MongoClient                # MongoDB bağlantısı
from langchain_openai import OpenAIEmbeddings  # Embedding modeli
from langchain_mongodb import MongoDBAtlasVectorSearch  # Vektör arama

# Kendi modüllerimiz
from app.agent.state import AgentState


# ============================================
# LOGGING AYARLARI
# ============================================
# Bu modül için özel logger oluştur
logger = logging.getLogger(__name__)


# ============================================
# ENVIRONMENT DEĞİŞKENLERİ
# ============================================
# .env dosyasını yükle
load_dotenv()

# --- API Anahtarı ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# --- MongoDB Ayarları ---
MONGO_URI: str = os.getenv("MONGO_URI", "")
DB_NAME: str = os.getenv("DB_NAME", "YasaaVisionDB")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "palmistry_knowledge")
INDEX_NAME: str = os.getenv("INDEX_NAME", "vector_index")

# --- Model Ayarları ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG Ayarları ---
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
"""
RAG_TOP_K: Kaç adet sonuç getirilecek?
- Düşük (3): Hızlı, az bağlam
- Yüksek (10): Yavaş, çok bağlam
- Önerilen: 5 (denge)
"""

# ============================================
# VECTOR STORE BAĞLANTISI
# ============================================
# Global değişkenler (lazy initialization için)
_vector_store: Optional[MongoDBAtlasVectorSearch] = None


def _get_vector_store() -> MongoDBAtlasVectorSearch:
    """
    MongoDB Atlas Vector Store bağlantısını döndürür.

    Lazy initialization kullanır - ilk çağrıda bağlantı kurulur,
    sonraki çağrılarda aynı instance döndürülür.

    Returns:
        MongoDBAtlasVectorSearch: Vector store instance

    Raises:
        ValueError: Gerekli environment değişkenleri eksikse
    """
    global _vector_store

    # Zaten bağlıysa, mevcut instance'ı döndür
    if _vector_store is not None:
        return _vector_store

    # Gerekli değişkenleri kontrol et
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY .env dosyasında bulunamadı!")

    if not MONGO_URI:
        raise ValueError("❌ MONGO_URI .env dosyasında bulunamadı!")

    logger.info(f"🔌 MongoDB'ye bağlanılıyor: {DB_NAME}/{COLLECTION_NAME}")

    # Embedding modeli oluştur
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    # MongoDB client ve collection
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    # Vector store oluştur
    _vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    logger.info("✅ MongoDB Vector Store bağlantısı kuruldu")
    return _vector_store


# ============================================
# SORGU HAZIRLAMA
# ============================================
def _prepare_search_query(vision_report: str) -> str:
    """
    Gözcü raporundan etkili bir arama sorgusu hazırlar.

    Gözcü'nün detaylı raporu bazen çok uzun olabilir.
    Bu fonksiyon raporu arama için optimize eder.

    Args:
        vision_report: Gözcü'nün teknik analiz raporu

    Returns:
        str: Optimize edilmiş arama sorgusu

    Example:
        >>> report = "Hand Shape: Square... Life Line: Deep and curved..."
        >>> query = _prepare_search_query(report)
        >>> # Sorgu artık anahtar terimlere odaklanmış
    """
    # Şimdilik raporu olduğu gibi kullan
    # İleride: Anahtar terimleri çıkarma, özetleme eklenebilir

    # Çok uzun raporları kırp (token limiti için)
    max_length = 1000  # Karakter limiti
    if len(vision_report) > max_length:
        logger.warning(f"   ⚠️ Rapor çok uzun ({len(vision_report)} karakter), kırpılıyor...")
        return vision_report[:max_length]

    return vision_report


# ============================================
# ANA NODE FONKSİYONU
# ============================================
def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Gözcü'nün raporuna göre veritabanında akademik bilgi arar.

    Bu fonksiyon LangGraph tarafından çağrılır.
    State'ten Gözcü raporunu alır, MongoDB'de arama yapar,
    bulunan dökümanları state'e ekler.

    Args:
        state: Mevcut graph state'i (AgentState)

    Returns:
        Dict[str, Any]: State güncellemeleri
            - retrieved_documents: Bulunan kitap sayfaları
            - error_message: Hata varsa mesaj

    Flow:
        1. State'den vision_analysis_report'u al
        2. Rapor yoksa boş döndür
        3. MongoDB'de similarity search yap
        4. Sonuçları state'e ekle

    Semantik Arama Nasıl Çalışır?
        1. Gözcü raporu: "Life line is deep and curved around Venus"
        2. Bu metin embedding'e çevrilir (1536 boyutlu vektör)
        3. MongoDB'deki tüm sayfa vektörleriyle karşılaştırılır
        4. En benzer K sayfa döndürülür (cosine similarity)
    """
    logger.info("--- 📚 ARAŞTIRMACI NODE: Kitaplar Taranıyor... ---")

    # ==========================================
    # ADIM 1: Gözcü Raporunu Al
    # ==========================================
    vision_report = state.get("visual_analysis_report")

    # Rapor yoksa - arama yapamayız
    if not vision_report:
        logger.warning("   ⚠️ Aranacak bir rapor yok, atlıyorum.")
        return {
            "retrieved_documents": [],
            "error_message": None
        }

    logger.info(f"   📝 Gözcü raporu alındı ({len(vision_report)} karakter)")

    # ==========================================
    # ADIM 2: Vector Store'u Hazırla
    # ==========================================
    try:
        vector_store = _get_vector_store()
    except ValueError as e:
        logger.error(f"   ❌ Vector store hatası: {e}")
        return {
            "retrieved_documents": [],
            "error_message": "Kitaplara erişirken bir sorun oluştu."
        }

    # ==========================================
    # ADIM 3: Arama Sorgusunu Hazırla
    # ==========================================
    search_query = _prepare_search_query(vision_report)

    # Log için sorgunun başını göster
    query_preview = search_query[:100].replace('\n', ' ')
    logger.info(f"   🔍 Arama sorgusu: '{query_preview}...'")

    # ==========================================
    # ADIM 4: Similarity Search Yap
    # ==========================================
    try:
        logger.info(f"   🔄 MongoDB'de arama yapılıyor (top_k={RAG_TOP_K})...")

        # Semantik arama - en benzer K dokümanı getir
        docs = vector_store.similarity_search(
            query=search_query,
            k=RAG_TOP_K
        )

        logger.info(f"   ✅ {len(docs)} adet sonuç bulundu")

    except Exception as e:
        logger.error(f"   ❌ Arama hatası: {e}")
        return {
            "retrieved_documents": [],
            "error_message": "Kitapları tararken bir hata oluştu, tekrar dener misin?"
        }

    # ==========================================
    # ADIM 5: Sonuçları İşle
    # ==========================================
    # Document objelerinden sadece içerikleri al
    retrieved_contents: List[str] = []

    for i, doc in enumerate(docs):
        # Her dokümanın kaynağını ve sayfa numarasını logla
        source = doc.metadata.get("source", "Bilinmeyen")
        page = doc.metadata.get("page", "?")

        logger.debug(f"   📖 Sonuç {i + 1}: {source} - Sayfa {page}")

        # İçeriği listeye ekle
        retrieved_contents.append(doc.page_content)

    # Sonuç özeti
    if retrieved_contents:
        logger.info(f"   📚 Toplam {len(retrieved_contents)} sayfa referans bulundu")
    else:
        logger.warning("   ⚠️ İlgili referans bulunamadı")

    return {
        "retrieved_documents": retrieved_contents,
        "error_message": None
    }


# ============================================
# TEST FONKSİYONU
# ============================================
def _test_retrieval_node():
    """
    Retrieval node'u test etmek için yardımcı fonksiyon.

    Kullanım:
        python -m App.agent.nodes.retrieval_node
    """
    print("=" * 50)
    print("📚 Retrieval Node Test")
    print("=" * 50)

    # Test için örnek bir state oluştur
    test_state: AgentState = {
        "messages": [],
        "user_image_bytes": None,
        "visual_analysis_report": "Life line is deep and curved around Mount of Venus. "
                                  "Head line is straight, ending near Mount of Moon. "
                                  "Heart line curves upward toward Mount of Jupiter.",
        "retrieved_documents": [],
        "final_response": None,
        "is_hand_detected": True,
        "error_message": None
    }

    print("\n📝 Test: Örnek Gözcü raporu ile arama")
    print(f"   Sorgu: {test_state['visual_analysis_report'][:80]}...")

    result = retrieval_node(test_state)

    print(f"\n📊 Sonuç:")
    print(f"   Bulunan döküman sayısı: {len(result.get('retrieved_documents', []))}")

    if result.get('retrieved_documents'):
        print(f"\n📖 İlk sonuç önizleme:")
        first_doc = result['retrieved_documents'][0]
        print(f"   {first_doc[:200]}...")

    print("\n" + "=" * 50)
    print("✅ Test tamamlandı!")
    print("=" * 50)


# ============================================
# MODÜL DOĞRUDAN ÇALIŞTIRILIRSA TEST YAP
# ============================================
if __name__ == "__main__":
    # Logging'i aktif et
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Testi çalıştır
    _test_retrieval_node()

































