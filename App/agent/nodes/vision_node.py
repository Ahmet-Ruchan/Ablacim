"""
============================================
YASAA VISION - Vision Analysis Node (Gözcü)
============================================
Bu düğüm, kullanıcının gönderdiği el fotoğrafını
GPT-4o Vision modeli ile analiz eder.

Görev:
- Fotoğrafın el olup olmadığını kontrol et
- El ise teknik analiz yap (çizgiler, tepeler, parmaklar)
- Kesinlikle YORUM yapma, sadece GÖZLEM yap

Çıktı:
- is_hand_detected: El tespit edildi mi?
- visual_analysis_report: Teknik analiz raporu

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os                                      # Environment değişkenleri için
import logging                                 # Profesyonel loglama
from typing import Dict, Any                   # Type hints için

from dotenv import load_dotenv                 # .env dosyası okuma
from langchain_openai import ChatOpenAI        # GPT-4o modeli
from langchain_core.messages import HumanMessage  # Mesaj formatı

# Kendi modüllerimiz
from App.agent.state import AgentState


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

import streamlit as st
def get_secret(name: str):
    return st.secrets.get(name) or os.getenv(name)


# --- API Anahtarı ---
OPENAI_API_KEY: str = get_secret("OPENAI_API_KEY")

# --- Model Ayarları ---
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
VISION_MAX_TOKENS: int = int(os.getenv("VISION_MAX_TOKENS", "1000"))


# ============================================
# MODEL BAŞLATMA
# ============================================
def _get_vision_llm() -> ChatOpenAI:
    """
    GPT-4o Vision modelini başlatır.

    Returns:
        ChatOpenAI: Yapılandırılmış model instance'ı

    Raises:
        ValueError: API key eksikse
    """
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY .env dosyasında bulunamadı!")

    return ChatOpenAI(
        model=VISION_MODEL,           # gpt-4o (vision destekli)
        api_key=OPENAI_API_KEY,       # API anahtarı
        max_tokens=VISION_MAX_TOKENS  # Maksimum çıktı uzunluğu
    )


# ============================================
# VISION PROMPT ŞABLONU (GÜÇLENDİRİLMİŞ)
# ============================================
# Bu prompt GPT-4o'ya "neye bakacağını" ve
# NASIL ifade edeceğini söyler
# DİKKAT: Kesin ifadeler kullan, "eğer/belki" YASAK!
VISION_ANALYSIS_PROMPT: str = """
**ROLE:** Expert Chiromancy (Palmistry) Morphologist with 30 years of experience.

**TASK:** Analyze the provided hand image and generate a HIGHLY DETAILED, DEFINITIVE technical report.

**STRICT LANGUAGE RULES (CRITICAL!):**
1. **BE 100% DEFINITIVE:** Do NOT use uncertain words like:
   - ❌ "appears to be", "might be", "looks like", "seems", "possibly", "could be", "if"
   - ✅ Use "IS", "ARE", "HAS", "SHOWS" - speak with absolute certainty
   - BAD: "The fingers appear to be conic shaped."
   - GOOD: "The fingers ARE Conic shaped with smooth, rounded tips and tapered ends."

2. **MAXIMUM DETAIL:** Do not just list features. Describe:
   - Exact measurements relative to palm
   - Texture and skin quality
   - Depth and width of every line
   - Specific angles and curves
   - Color variations if visible

3. **NO INTERPRETATIONS:** Only physical descriptions. No meanings, no advice.

**COMPREHENSIVE ANALYSIS CHECKLIST:**

## 1. HAND SHAPE (Be specific about WHY)
- Exact classification: Square, Spatulate, Philosophic, Conic, Psychic, Elementary, or Mixed
- Palm width compared to length (ratio)
- Overall hand size relative to body (if visible)
- Flesh consistency: Soft/Flabby, Medium/Elastic, Hard/Firm

## 2. FINGERS (Each finger individually)
- **Length:** Relative to palm length, relative to each other
- **Tip Shapes:** Square, Pointed/Conic, Spatulate, Mixed
- **Joints:** Smooth or Knotty (Philosophic knots vs Practical knots)
- **Setting on Palm:** Even line, arch shape, or irregular
- **Spaces Between:** When held naturally - wide gaps or close together
- **THUMB (Critical):**
  - Setting: High, Medium, or Low on palm
  - Flexibility: Stiff (unbending) or Supple (bends back easily)
  - First Phalange (Will) vs Second Phalange (Logic) ratio
  - Angle of opening from hand

## 3. MAJOR LINES (Extremely detailed)

**LIFE LINE:**
- Starting point: Exact location between thumb and index finger
- Path: Close to thumb, wide curve around Venus, or moderate
- Ending point: Where exactly does it terminate?
- Depth: Deep/Medium/Faint/Chained
- Width: Broad or Fine
- Special Marks: Islands, breaks, chains, branches, crosses, stars
- Sister Line present? (Mars Line)

**HEAD LINE:**
- Starting point: Joined with Life Line? Separated? How much gap?
- Direction: Straight across palm, sloping toward Moon, or rising
- Length: Reaches Mercury? Stops at Apollo? Short?
- Ending: Clean end, fork (Writer's Fork), multiple branches
- Depth and clarity throughout its length
- Special marks: Islands (concentration issues), breaks, chains

**HEART LINE:**
- Starting point: Under Mercury finger
- Termination: Under Jupiter, between Jupiter-Saturn, under Saturn, or forked
- Curvature: Straight, curved upward, deeply curved
- Depth: Deep (passionate), Medium, Faint (reserved)
- Branches: Upward branches, downward branches, clean
- Girdle of Venus present above it?

**FATE LINE (if present):**
- Starting point: Wrist, Life Line, Moon mount, or middle of palm
- Path: Straight, curved, broken, multiple lines
- Ending point: Saturn, Jupiter, or other

## 4. MOUNTS (Rate each: Flat/Normal/Raised/Padded/Overdeveloped)
- **Venus** (base of thumb): Size, firmness, boundaries
- **Jupiter** (under index): Elevation, size
- **Saturn** (under middle): Presence, development
- **Apollo/Sun** (under ring): Prominence
- **Mercury** (under little): Development
- **Moon/Luna** (opposite thumb, lower): Size, padding
- **Mars Positive** (under Jupiter, inner palm)
- **Mars Negative** (under Mercury, inner palm)
- **Plain of Mars** (center of palm): Hollow or filled

## 5. SKIN TEXTURE & ADDITIONAL FEATURES
- Skin quality: Fine/Silky, Medium, Coarse/Rough
- Line density: Many fine lines (sensitive) or few main lines (simple nature)
- Color: Pink, pale, red, yellow tones
- Nails (if visible): Shape, moons, ridges

**OUTPUT FORMAT:**
Write a DENSE, CONTINUOUS technical narrative of approximately 400-500 words.
Do NOT use bullet points or headers in your output.
Write it as flowing professional prose, as if dictating a medical report.
Every statement must be DEFINITIVE - you are the expert, speak with authority.

**IMPORTANT - HAND DETECTION RULES:**
- ONLY respond with "NOT_A_HAND" if the image clearly shows something completely different (like a car, building, animal, text document)
- If you can see ANY hand or palm features AT ALL, even partially or at an angle, PROCEED WITH ANALYSIS
- If image quality is poor but it's a hand, do your best and note "LOW_QUALITY" at the start
- When in doubt, ANALYZE - err on the side of providing analysis rather than rejecting
- Hands photographed at angles, with objects in background, or partially visible should still be analyzed
"""


# ============================================
# ANA NODE FONKSİYONU
# ============================================
def vision_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Kullanıcının gönderdiği el fotoğrafını analiz eder.

    Bu fonksiyon LangGraph tarafından çağrılır.
    State'i alır, görsel analiz yapar, sonuçları döndürür.

    Args:
        state: Mevcut graph state'i (AgentState)

    Returns:
        Dict[str, Any]: State güncellemeleri
            - is_hand_detected: El tespit edildi mi?
            - visual_analysis_report: Teknik rapor (veya None)
            - error_message: Hata mesajı (veya None)

    Flow:
        1. State'den resim verisini al
        2. Resim yoksa atla
        3. GPT-4o'ya gönder
        4. Sonucu parse et
        5. State güncellemelerini döndür
    """
    logger.info("--- 👁️ GÖZCÜ NODE: Fotoğraf Analiz Ediliyor... ---")

    # ==========================================
    # ADIM 1: Resim Verisini Al
    # ==========================================
    image_data = state.get("user_image_bytes")

    # Resim yoksa - kullanıcı sadece sohbet ediyor olabilir
    if not image_data:
        logger.warning("   ⚠️ Resim bulunamadı, görsel analiz atlanıyor.")
        return {
            "is_hand_detected": False,
            "visual_analysis_report": None,
            "error_message": None  # Bu bir hata değil, sadece resim yok
        }

    logger.info(f"   📸 Resim verisi alındı ({len(image_data)} karakter)")

    # ==========================================
    # ADIM 2: GPT-4o Vision'ı Hazırla
    # ==========================================
    try:
        llm = _get_vision_llm()
        logger.info(f"   🤖 Model yüklendi: {VISION_MODEL}")
    except ValueError as e:
        logger.error(f"   ❌ Model yükleme hatası: {e}")
        return {
            "is_hand_detected": False,
            "visual_analysis_report": None,
            "error_message": "Sistem hatası oluştu, lütfen tekrar deneyin."
        }

    # ==========================================
    # ADIM 3: Mesajı Hazırla ve Gönder
    # ==========================================
    # LangChain formatında multimodal mesaj oluştur
    message = HumanMessage(
        content=[
            # Metin kısmı: Prompt
            {
                "type": "text",
                "text": VISION_ANALYSIS_PROMPT
            },
            # Görsel kısmı: Base64 encoded resim
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            },
        ]
    )

    # ==========================================
    # ADIM 4: API Çağrısı
    # ==========================================
    try:
        logger.info("   🔄 GPT-4o Vision API çağrısı yapılıyor...")
        response = llm.invoke([message])
        analysis = response.content
        logger.info("   ✅ API yanıtı alındı")

    except Exception as e:
        # API hatası (rate limit, network, vb.)
        logger.error(f"   ❌ Vision API hatası: {e}")
        return {
            "is_hand_detected": False,
            "visual_analysis_report": None,
            "error_message": "Fotoğrafı analiz edemedim, tekrar dener misin kuzum?"
        }

    # ==========================================
    # ADIM 5: Sonucu Değerlendir
    # ==========================================

    # Durum 1: El değil - SADECE cevap çok kısa ve NOT_A_HAND içeriyorsa
    # Bu, model'in uzun bir analizde bu kelimeyi kullanmasını engelliyor
    analysis_stripped = analysis.strip()
    is_rejection = (
        "NOT_A_HAND" in analysis_stripped and
        len(analysis_stripped) < 100  # Kısa cevaplar = gerçek red
    )

    if is_rejection:
        logger.warning("   ❌ Gönderilen fotoğraf el değil")
        return {
            "is_hand_detected": False,
            "visual_analysis_report": None,
            "error_message": "Kuzum bu el fotoğrafı değil gibi görünüyor. "
                           "Avuç içini düzgünce gösteren bir fotoğraf atar mısın?"
        }

    # Durum 2: Düşük kalite ama el
    if "LOW_QUALITY" in analysis:
        logger.warning("   ⚠️ Düşük kaliteli el fotoğrafı")
        # Yine de analize devam et, ama not düş
        analysis = analysis + "\n\n[NOT: Fotoğraf kalitesi düşük, analiz kısıtlı olabilir]"

    # Durum 3: Başarılı analiz
    logger.info("   ✅ El fotoğrafı başarıyla analiz edildi")
    logger.debug(f"   📝 Analiz önizleme: {analysis[:200]}...")

    return {
        "is_hand_detected": True,
        "visual_analysis_report": analysis,
        "error_message": None
    }


# ============================================
# TEST FONKSİYONU
# ============================================
def _test_vision_node():
    """
    Vision node'u test etmek için yardımcı fonksiyon.

    Kullanım:
        python -m App.agent.nodes.vision_node
    """
    import base64

    print("=" * 50)
    print("👁️ Vision Node Test")
    print("=" * 50)

    # Test için örnek bir state oluştur (resim olmadan)
    test_state: AgentState = {
        "messages": [],
        "user_image_bytes": None,  # Test için resim yok
        "visual_analysis_report": None,
        "retrieved_documents": [],
        "final_response": None,
        "is_hand_detected": False,
        "error_message": None
    }

    print("\n📝 Test 1: Resim olmadan çağır")
    result = vision_analysis_node(test_state)
    print(f"   Sonuç: {result}")

    print("\n" + "=" * 50)
    print("✅ Test tamamlandı!")
    print("=" * 50)
    print("\n💡 Gerçek bir el fotoğrafı ile test etmek için:")
    print("   1. Bir el fotoğrafını base64'e çevirin")
    print("   2. test_state['user_image_bytes'] = base64_data")
    print("   3. vision_analysis_node(test_state) çağırın")


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
    _test_vision_node()