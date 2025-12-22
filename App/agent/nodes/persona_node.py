"""
============================================
Boncuk VISION - Persona Node (Abla)
============================================
Bu düğüm, toplanan tüm verileri alır ve
"Boncuk Abla" personası ile kullanıcıya sunar.

Görev:
- Gözcü'nün teknik raporunu al
- Araştırmacı'nın kitap referanslarını al
- Bunları sıcak, samimi "Abla" üslubuyla yorumla
- Türkçe cevap üret

Persona Özellikleri:
- Sıcak ve samimi ("Kuzum", "Aslanım", "Canım")
- Bilgili ama ulaşılabilir
- Sandviç Tekniği: Övgü → Uyarı → Motivasyon
- Referanslı (Kitaplardan alıntı yapar)

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os  # Environment değişkenleri için
import logging  # Profesyonel loglama
from typing import Dict, Any, List  # Type hints için

from dotenv import load_dotenv  # .env dosyası okuma
from langchain_openai import ChatOpenAI  # GPT-4o modeli
from langchain_core.messages import (  # Mesaj formatları
    SystemMessage,
    HumanMessage
)

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
PERSONA_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")  # Aynı model
PERSONA_MAX_TOKENS: int = int(os.getenv("PERSONA_MAX_TOKENS", "1500"))
"""
PERSONA_MAX_TOKENS: Abla'nın cevap uzunluğu
- 1000: Kısa, öz yorumlar
- 1500: Orta detay (önerilen)
- 2000: Uzun, detaylı fallar
"""


# ============================================
# MODEL BAŞLATMA
# ============================================
def _get_persona_llm() -> ChatOpenAI:
    """
    Abla persona için GPT-4o modelini başlatır.

    Returns:
        ChatOpenAI: Yapılandırılmış model instance'ı

    Raises:
        ValueError: API key eksikse
    """
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY .env dosyasında bulunamadı!")

    return ChatOpenAI(
        model=PERSONA_MODEL,
        api_key=OPENAI_API_KEY,
        max_tokens=PERSONA_MAX_TOKENS,
        temperature=0.8  # Biraz yaratıcılık için
    )


# ============================================
# ABLA PERSONA PROMPT'U
# ============================================
ABLA_SYSTEM_PROMPT: str = """
Sen 'Boncuk Abla'sın. Geleneksel Türk falcı ağzıyla konuşan, hem mistik hem de bilimsel el falı (Kiromansi) bilen bir yapay zeka asistanısın.

## 🎭 KİMLİĞİN
- Adın: Boncuk Abla
- Uzmanlık: Kiromansi (El Falı), özellikle Benham ve St. Germain ekolü
- Deneyim: Yıllardır el okuyan, binlerce ele bakan tecrübeli bir falcı

## 🗣️ TONUN VE ÜSLUBUN
1. **Samimi ve Sıcak:**
   - "Kuzum", "Aslanım", "Canımın içi", "Güzel yavrum" gibi hitaplar kullan
   - Mesafeli değil, sanki karşında tanıdığın biri varmış gibi konuş

2. **Bilgili ama Ulaşılabilir:**
   - Teknik terimleri kullan ama açıkla
   - "Hayat çizgin" de, "Life Line" deme
   - Kitaplardan referans ver: "Benham kitabında da yazar ki..."

3. **Sandviç Tekniği (ÖNEMLİ!):**
   - 🍞 ÖNCE: Güzel bir özelliğinden bahset (övgü)
   - 🥬 SONRA: Dikkat etmesi gereken noktayı söyle (uyarı)
   - 🍞 EN SON: Motive edici bir cümleyle bitir (umut)

4. **Dobra ama Kırıcı Değil:**
   - Kötü bir şey görsen bile yıkıcı olma
   - "Şurada biraz zorluk var ama..." şeklinde yumuşat
   - Asla "Başına kötü şeyler gelecek" gibi kehanetlerde bulunma

5. **Mistik Hava:**
   - Ara sıra "Maşallah", "İnşallah", "Allah korusun" gibi ifadeler kullan
   - Ama batıl inançlara değil, gözleme dayalı konuş

## ⚠️ YAPMAMAN GEREKENLER
- Asla "Ben bir yapay zekayım" deme
- Kesin tarih veya isim verme ("2024'te evleneceksin" ❌)
- Sağlık teşhisi koyma ("Kalp hastalığın var" ❌)
- Ölüm, kaza gibi korkutucu kehanetler yapma
- İngilizce terim kullanma (Head Line → Akıl Çizgisi)

## 📝 CEVAP FORMATI
1. Kısa bir selamlama
2. Elin genel değerlendirmesi (el tipi)
3. Çizgilerin yorumu (en az 3 ana çizgi)
4. Tepelerin/dağların yorumu
5. Genel değerlendirme ve tavsiyeler
6. Motive edici kapanış

## 🌍 DİL
- Sana gelen veriler İNGİLİZCE olacak (teknik analiz)
- Sen bunları TÜRKÇE yorumlayacaksın
- Akıcı, doğal Türkçe kullan
"""


# ============================================
# KULLANICI İÇERİĞİ ŞABLONU
# ============================================
def _build_user_content(
        vision_report: str,
        book_references: List[str]
) -> str:
    """
    Abla'ya gönderilecek kullanıcı içeriğini oluşturur.

    Args:
        vision_report: Gözcü'nün teknik analiz raporu
        book_references: Kitaplardan bulunan referanslar

    Returns:
        str: Formatlanmış kullanıcı içeriği
    """
    # Kitap referanslarını birleştir
    if book_references:
        references_text = "\n\n---\n\n".join(book_references)
    else:
        references_text = "Kitaplarda bu özellikler hakkında spesifik bir referans bulunamadı. Genel bilginle yorum yap."

    # Şablonu doldur
    content = f"""
## 📋 GÖZCÜ'NÜN TEKNİK ANALİZİ (İngilizce)
{vision_report}

## 📚 KİTAPLARDAN BULUNAN REFERANSLAR
{references_text}

---

Yukarıdaki teknik verileri ve kitap referanslarını kullanarak, bu kişinin elini Boncuk Abla olarak yorumla.
Sandviç tekniğini unutma: Övgü → Uyarı → Motivasyon
"""

    return content


# ============================================
# ANA NODE FONKSİYONU
# ============================================
def persona_node(state: AgentState) -> Dict[str, Any]:
    """
    Toplanan tüm teknik verileri 'Abla' personasıyla kullanıcıya sunar.

    Bu fonksiyon LangGraph tarafından çağrılır.
    Gözcü raporu ve kitap referanslarını alır,
    sıcak ve samimi bir Türkçe yorum üretir.

    Args:
        state: Mevcut graph state'i (AgentState)

    Returns:
        Dict[str, Any]: State güncellemeleri
            - final_response: Abla'nın Türkçe yorumu
            - error_message: Hata varsa mesaj

    Flow:
        1. State'den vision_report ve retrieved_documents al
        2. System prompt (Abla personası) hazırla
        3. User content (teknik veri + referanslar) hazırla
        4. GPT-4o'ya gönder
        5. Türkçe yorumu state'e ekle
    """
    logger.info("--- 🗣️ ABLA NODE: Fal Yazılıyor... ---")

    # ==========================================
    # ADIM 1: Verileri Al
    # ==========================================
    vision_report = state.get("visual_analysis_report", "")
    book_references = state.get("retrieved_documents", [])

    # Kontrol: En azından gözcü raporu olmalı
    if not vision_report:
        logger.error("   ❌ Gözcü raporu bulunamadı!")
        return {
            "final_response": None,
            "error_message": "Kuzum, elini göremedim ki falına bakayım. "
                             "Bir el fotoğrafı atar mısın?"
        }

    logger.info(f"   📝 Gözcü raporu: {len(vision_report)} karakter")
    logger.info(f"   📚 Kitap referansı: {len(book_references)} adet")

    # ==========================================
    # ADIM 2: Modeli Hazırla
    # ==========================================
    try:
        llm = _get_persona_llm()
        logger.info(f"   🤖 Model yüklendi: {PERSONA_MODEL}")
    except ValueError as e:
        logger.error(f"   ❌ Model yükleme hatası: {e}")
        return {
            "final_response": None,
            "error_message": "Ay kuzum, dilim tutuldu bir anlık. Tekrar dener misin?"
        }

    # ==========================================
    # ADIM 3: Mesajları Hazırla
    # ==========================================
    # System message: Abla personası
    system_message = SystemMessage(content=ABLA_SYSTEM_PROMPT)

    # User message: Teknik veri + Referanslar
    user_content = _build_user_content(vision_report, book_references)
    user_message = HumanMessage(content=user_content)

    messages = [system_message, user_message]

    logger.debug(f"   📨 User content uzunluğu: {len(user_content)} karakter")

    # ==========================================
    # ADIM 4: API Çağrısı
    # ==========================================
    try:
        logger.info("   🔄 Abla düşünüyor...")
        response = llm.invoke(messages)
        abla_response = response.content
        logger.info("   ✅ Fal yorumu hazırlandı")

    except Exception as e:
        logger.error(f"   ❌ API hatası: {e}")
        return {
            "final_response": None,
            "error_message": "Kuzum nazar değdi galiba, dilim bağlandı. "
                             "Bir dakika sonra tekrar dener misin?"
        }

    # ==========================================
    # ADIM 5: Sonucu Döndür
    # ==========================================
    # Cevabın uzunluğunu logla
    logger.info(f"   📜 Yorum uzunluğu: {len(abla_response)} karakter")

    return {
        "final_response": abla_response,
        "error_message": None
    }


# ============================================
# TEST FONKSİYONU
# ============================================
def _test_persona_node():
    """
    Persona node'u test etmek için yardımcı fonksiyon.

    Kullanım:
        python -m App.agent.nodes.persona_node
    """
    print("=" * 50)
    print("🗣️ Persona Node (Abla) Test")
    print("=" * 50)

    # Test için örnek bir state oluştur
    test_state: AgentState = {
        "messages": [],
        "user_image_bytes": None,
        "visual_analysis_report": """
        HAND SHAPE: Square type based on equal palm width and finger length.

        PRIMARY LINES:
        - Life Line: Deep and widely curved around Mount of Venus. No breaks or islands.
        - Head Line: Straight, medium length, ending near Mount of Moon. Slight fork at end.
        - Heart Line: Curved upward, terminating under Mount of Jupiter. Deep and clear.

        MOUNTS:
        - Mount of Venus: Padded (prominent)
        - Mount of Jupiter: Raised
        - Mount of Moon: Normal

        FINGERS:
        - Thumb setting: Medium
        - Finger tips: Square
        """,
        "retrieved_documents": [
            "--- PAGE 145 ---\nA deep Life line indicates vitality and robust health...",
            "--- PAGE 203 ---\nWhen the Heart line ends under Jupiter, it shows idealistic love..."
        ],
        "final_response": None,
        "is_hand_detected": True,
        "error_message": None
    }

    print("\n📝 Test: Örnek veri ile fal yorumu üretme")
    print("   (Bu test gerçek API çağrısı yapar, maliyet oluşabilir)")

    user_input = input("\n   Devam etmek istiyor musun? (e/h): ")

    if user_input.lower() == 'e':
        result = persona_node(test_state)

        if result.get('final_response'):
            print("\n" + "=" * 50)
            print("🔮 ABLA'NIN YORUMU:")
            print("=" * 50)
            print(result['final_response'])
        else:
            print(f"\n❌ Hata: {result.get('error_message')}")
    else:
        print("   Test atlandı.")

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
    _test_persona_node()