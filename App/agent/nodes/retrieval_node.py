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



































