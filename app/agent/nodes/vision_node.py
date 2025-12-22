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

# --- Model Ayarları ---
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
VISION_MAX_TOKENS: int = int(os.getenv("VISION_MAX_TOKENS", "1000"))


# ============================================
# MODEL BAŞLATMA
# ============================================

def _get_vision_llm() -> ChatOpenAI:

    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY .env dosyasında bulunamadı!")

    return ChatOpenAI(
        model=VISION_MODEL,
        api_key=OPENAI_API_KEY,
        max_tokens=VISION_MAX_TOKENS
    )


# ============================================
# VISION PROMPT ŞABLONU
# ============================================
# Bu prompt GPT-4o'ya "neye bakacağını" söyler
# DİKKAT: Yorum yapma, sadece gözlem yap!

VISION_ANALYSIS_PROMPT: str = """
**ROLE:** Expert Chiromancy (Palmistry) Morphologist.

**TASK:** Analyze the user's hand image accurately and objectively.

**OUTPUT FORMAT:**
Please extract and describe these technical details in a structured way:

1. **HAND SHAPE:**
   - Type: (Square, Spatulate, Conic, Psychic, Philosophic, Elementary, Mixed)
   - Reasoning: (Based on palm width vs finger length ratio)

2. **PRIMARY LINES:**
   - **Life Line:** 
     * Length: (Long/Medium/Short)
     * Depth: (Deep/Medium/Faint)  
     * Curvature: (Widely curved around Venus / Straight / Close to thumb)
     * Special marks: (Islands, breaks, forks, branches - if any)

   - **Head Line:**
     * Direction: (Sloping toward Moon / Straight across / Rising toward fingers)
     * Length: (Reaches Mercury / Stops at Apollo / Short)
     * Fork at end: (Yes/No)

   - **Heart Line:**
     * Termination: (Under Jupiter / Between Jupiter-Saturn / Under Saturn)
     * Curvature: (Curved upward / Straight / Curved downward)
     * Depth: (Deep/Medium/Faint)

3. **MOUNTS (Prominence Level: Flat/Normal/Raised/Padded):**
   - Mount of Venus (thumb base)
   - Mount of Jupiter (under index finger)
   - Mount of Saturn (under middle finger)
   - Mount of Apollo (under ring finger)
   - Mount of Mercury (under little finger)
   - Mount of Moon (opposite thumb, lower palm)

4. **FINGERS:**
   - Thumb setting: (High/Medium/Low on palm)
   - Finger tips: (Pointed/Conic/Square/Spatulate)
   - Notable features: (Long/short fingers, gaps between fingers)

**CRITICAL INSTRUCTIONS:**
- Do NOT interpret meanings (e.g., "You will be rich", "You will travel")
- Do NOT give advice or predictions
- ONLY describe physical features you observe
- If the image is NOT a clear hand photo, respond with exactly: "NOT_A_HAND"
- If image quality is poor but it's a hand, do your best and note "LOW_QUALITY"
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

    if not image_data:
        logger.warning("   ⚠️ Resim bulunamadı, görsel analiz atlanıyor.")
        return {
            "is_hand_detected": False,
            "visual_analysis_report": None,
            "error_message": None
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
                    "ur": f"data:image/jpeg;base64,{image_data}"
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

    # Durum 1: El değil
    if "NOT_A_HAND" in analysis:
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
        python -m app.agent.nodes.vision_node
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


