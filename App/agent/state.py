"""
============================================
YASAA VISION - Graph State Definition
============================================
Bu dosya, LangGraph içindeki node'lar (düğümler) arasında
taşınacak olan veri paketini (State) tanımlar.

Düğümler:
- Gözcü (Vision Node): El fotoğrafını analiz eder
- Araştırmacı (Retrieval Node): MongoDB'den bilgi çeker
- Abla (Persona Node): Son cevabı oluşturur

Her düğüm bu state'i okur, işler ve günceller.
Bir nevi "koşucu" gibi düşün - her düğüm topu alır,
kendi işini yapar ve bir sonrakine paslar.
============================================
"""

from typing import (
    TypedDict,
    List,
    Optional,
    Annotated
)

from langchain_core.messages import BaseMessage











































