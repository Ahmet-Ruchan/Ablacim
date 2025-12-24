"""
============================================
YASAA VISION - Persona Node (Abla)
============================================
Bu düğüm, toplanan tüm verileri alır ve
"Yasaa Abla" personası ile kullanıcıya sunar.

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
import os                                      # Environment değişkenleri için
import logging                                 # Profesyonel loglama
from typing import Dict, Any, List             # Type hints için

from dotenv import load_dotenv                 # .env dosyası okuma
from langchain_openai import ChatOpenAI        # GPT-4o modeli
from langchain_core.messages import (          # Mesaj formatları
    SystemMessage,
    HumanMessage
)

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
#OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY","")

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
# ABLA PERSONA PROMPT'U (GÜÇLENDİRİLMİŞ)
# ============================================
# Bu prompt Abla'ya:
# 1. Kesin ifadeler kullanmayı (EĞER YASAK!)
# 2. Başlık atmamayı
# 3. Uzun ve detaylı cevap vermeyi
# 4. El raporuna TAM BAĞLILIK ile konuşmayı öğretir
ABLA_SYSTEM_PROMPT: str = """
Sen 'Yasaa Abla'sın. Profesyonel, mistik ve bilimsel el falı (Kiromansi) uzmanısın.

## 🎭 KİMLİĞİN
- Adın: Yasaa Abla
- Uzmanlık: Kiromansi (El Falı), özellikle Benham ve St. Germain ekolü
- Deneyim: 30 yıldır el okuyan, on binlerce ele bakan usta bir falcı

## ⛔ KESİN YASAKLAR (BUNLARA UYMAZSAN SİSTEM ÇÖKER!)

### YASAK 1: "EĞER" KELİMESİ VE ŞART KİPİ
❌ ASLA şunları söyleme:
- "Eğer elin kareyse..."
- "Eğer hayat çizgin uzunsa..."
- "Şayet parmakların sivri ise..."
- "...olabilir", "...görünüyor", "...gibi duruyor"

✅ BUNUN YERİNE kesin konuş:
- "Senin elin KARE. Bu demek ki..."
- "Hayat çizgin ÇOK UZUN ve DERİN. Bu gösteriyor ki..."
- "Parmakların SİVRİ UÇLU. Yani sen..."

⚠️ NEDEN? Çünkü sana gelen teknik rapor zaten elin nasıl olduğunu söylüyor. 
Sen varsayım yapmıyorsun, GÖRDÜĞÜNÜ söylüyorsun. Falcı dediğin "Görüyorum ki böylesin!" der.

### YASAK 2: BAŞLIK VE BÖLÜM ATLAMA
❌ ASLA şunları yazma:
- "Giriş:", "Sonuç:", "Özet:"
- "1.", "2.", "3." gibi numaralı listeler
- "Sandviç Tekniği", "Övgü Bölümü", "Uyarı Bölümü"
- "El Analizi:", "Kariyer:", "Aşk:" gibi başlıklar
- Bullet point veya madde işaretleri

✅ BUNUN YERİNE akıcı bir sohbet yaz:
Sanki karşında oturan birine konuşur gibi, paragraflar halinde, doğal bir dille anlat.
Bir arkadaşına anlatır gibi yaz, akademik makale gibi değil.

### YASAK 3: KISA CEVAP
❌ 2-3 paragrafla bitirme
✅ EN AZ 5-6 paragraf, doyurucu ve detaylı bir analiz yap

## 🎯 ANA GÖREVİN
Kullanıcının EL ANALİZİNDEKİ GERÇEK bulguları (el tipi, çizgiler, tepeler, parmaklar) kullanarak, 
kullanıcının SORUSUNA kişiselleştirilmiş, UZUN ve DETAYLI cevap ver.

**STRATEJİN:**

1. **Rapordaki Gerçekleri Kullan:**
   - Rapor "Square hand" diyorsa → "Senin elin KARE TİPİNDE" de
   - Rapor "Life line is deep" diyorsa → "Hayat çizgin ÇOK DERİN" de
   - Asla tahmin yapma, raporda NE YAZIYORSA onu söyle

2. **Her Bulguyu Yorumla:**
   - Sadece "Akıl çizgin uzun" deme
   - "Akıl çizgin uzun, bu senin analitik düşünce yeteneğinin güçlü olduğunu gösteriyor. Benham'ın 'The Laws of Scientific Hand Reading' kitabında da belirttiği gibi, uzun akıl çizgisi olan insanlar detaylara dikkat eder ve mantıklı kararlar alır. Senin için bu demek oluyor ki..." şeklinde AÇIKLA

3. **Soruyla Bağlantı Kur:**
   - Kullanıcı "Kariyer" sorduysa → Akıl Çizgisi, Parmak Şekli, Başparmak gücünden yola çık
   - Kullanıcı "Aşk/Evlilik" sorduysa → Kalp Çizgisi, Venüs Tepesi'nden yola çık
   - Her zaman NEDEN bu sonuca vardığını eldeki işaretlerle kanıtla

4. **Karakter Analizi Yap:**
   - El bulgularından kişilik özelliklerini çıkar
   - Bu özelliklerin sorulan konuyla ilişkisini ayrıntılı kur

## 🗣️ TONUN VE ÜSLUBUN
1. **Samimi ve Sıcak ama Otoriter:**
   - "Bak kuzum", "Dinle beni", "Şimdi sana bir şey söyleyeceğim"
   - "Kuzum", "Aslanım", "Canımın içi", "Güzel yavrum"
   - Sanki yıllardır tanıdığın birine konuşur gibi

2. **Kesin ve Net Konuş:**
   - "Görüyorum ki sen...", "Elin bana diyor ki...", "Bu çizgi açıkça gösteriyor..."
   - Tereddüt yok, sen uzmansın, gördüğünü söylüyorsun

3. **Referans Ver:**
   - "Benham kitabının şu bölümünde de yazar ki..."
   - "Kiromansi biliminde bu işaret şu anlama gelir..."

4. **Akıcı Sohbet:**
   - Paragraflar arası geçişler doğal olsun
   - Bir konudan diğerine akıcı geç
   - Sonunda motive edici bir kapanış yap

## ⚠️ DİĞER YASAKLAR
- Asla "Ben bir yapay zekayım" deme
- Kesin tarih veya isim verme ("2024'te evleneceksin" ❌)
- Sağlık teşhisi koyma ("Kalp hastalığın var" ❌)
- Ölüm, kaza gibi korkutucu kehanetler yapma
- İngilizce terim kullanma (Head Line → Akıl Çizgisi)
- Soruyu görmezden gelip sadece genel el yorumu yapma

## 🌍 DİL
- Sana gelen veriler İNGİLİZCE olacak (teknik analiz)
- Sen bunları TÜRKÇE yorumlayacaksın
- Akıcı, doğal, samimi Türkçe kullan
- EN AZ 1500-2000 kelime uzunluğunda cevap ver
"""


# ============================================
# KULLANICI SORUSUNU ÇIKARMA
# ============================================
def _extract_user_question(messages: list) -> str:
    """
    State'deki messages listesinden kullanıcının sorusunu çıkarır.

    Args:
        messages: State'deki mesaj listesi

    Returns:
        str: Kullanıcının sorusu veya varsayılan metin

    Desteklenen formatlar:
    - HumanMessage objesi
    - Tuple: ("user", "soru metni")
    - Dict: {"role": "user", "content": "soru metni"}
    """
    # Varsayılan: Soru yoksa genel yorum iste
    default_question = "Genel bir el falı yorumu istiyorum."

    # Mesaj listesi boşsa
    if not messages or len(messages) == 0:
        return default_question

    # Son mesajı al (en güncel soru)
    last_message = messages[-1]

    # Format 1: LangChain HumanMessage objesi
    if hasattr(last_message, 'content') and last_message.content:
        return last_message.content

    # Format 2: Tuple ("user", "soru metni")
    if isinstance(last_message, tuple) and len(last_message) >= 2:
        role, content = last_message[0], last_message[1]
        if role == "user" and content:
            return content

    # Format 3: Dict {"role": "user", "content": "soru metni"}
    if isinstance(last_message, dict):
        if last_message.get("role") == "user" and last_message.get("content"):
            return last_message["content"]

    return default_question


# ============================================
# SOHBET GEÇMİŞİNİ METİNE DÖNÜŞTÜRME
# ============================================
def _build_chat_history_text(messages: list) -> str:
    """
    State'deki mesaj listesini okunabilir metin formatına çevirir.

    Args:
        messages: State'deki mesaj listesi

    Returns:
        str: Formatlanmış sohbet geçmişi

    Bu fonksiyon Abla'nın önceki konuşmaları hatırlamasını sağlar.
    Son 6 mesajı alır ki context window dolmasın.
    """
    from langchain_core.messages import HumanMessage as HM, AIMessage as AM

    if not messages or len(messages) == 0:
        return "Bu ilk konuşmamız."

    chat_lines = []

    # Son 6 mesajı al (hafıza için yeterli, token için güvenli)
    recent_messages = messages[-6:]

    for msg in recent_messages:
        # LangChain HumanMessage
        if isinstance(msg, HM) or (hasattr(msg, '__class__') and msg.__class__.__name__ == 'HumanMessage'):
            chat_lines.append(f"Kullanıcı: {msg.content}")
        # LangChain AIMessage
        elif isinstance(msg, AM) or (hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage'):
            # Abla'nın cevabını kısalt (çok uzun olabilir)
            short_response = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            chat_lines.append(f"Abla: {short_response}")
        # Tuple format
        elif isinstance(msg, tuple) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            if role == "user":
                chat_lines.append(f"Kullanıcı: {content}")
            else:
                short_content = content[:300] + "..." if len(content) > 300 else content
                chat_lines.append(f"Abla: {short_content}")

    return "\n".join(chat_lines) if chat_lines else "Bu ilk konuşmamız."


# ============================================
# KULLANICI İÇERİĞİ ŞABLONU (GÜÇLENDİRİLMİŞ)
# ============================================
def _build_user_content(
    vision_report: str,
    book_references: List[str],
    user_question: str,
    chat_history: str = ""
) -> str:
    """
    Abla'ya gönderilecek kullanıcı içeriğini oluşturur.

    Artık şunları içeriyor:
    - Sohbet geçmişi (hafıza)
    - Kullanıcı sorusu
    - Teknik analiz raporu
    - Kitap referansları
    - Güçlendirilmiş talimatlar

    Args:
        vision_report: Gözcü'nün teknik analiz raporu
        book_references: Kitaplardan bulunan referanslar
        user_question: Kullanıcının sorusu
        chat_history: Önceki sohbet geçmişi (opsiyonel)

    Returns:
        str: Formatlanmış kullanıcı içeriği
    """
    # Kitap referanslarını birleştir
    if book_references:
        references_text = "\n\n---\n\n".join(book_references)
    else:
        references_text = "Kitaplarda bu özellikler hakkında spesifik referans bulunamadı. Genel kiromansi bilginle yorum yap."

    # Şablonu doldur
    content = f"""
## 📜 SOHBET GEÇMİŞİ (Önceki konuşmalarınız - BAĞLAMI KORU!)
{chat_history}

---

## 🎯 KULLANICININ ŞU ANKİ SORUSU
"{user_question}"

---

## 📋 EL ANALİZ RAPORU (KESİN VERİ - BUNA TAM BAĞLI KAL!)
{vision_report}

---

## 📚 AKADEMİK KANITLAR (Kitaplardan)
{references_text}

---

## ⚠️ KRİTİK TALİMATLAR (MUTLAKA UYULMALI!)

1. **EĞER KULLANMA:** Raporda "Square hand" yazıyorsa "Senin elin KARE" de, "Eğer elin kareyse" DEME!

2. **BAŞLIK ATMA:** "Giriş:", "Sonuç:", "1.", "2." gibi başlıklar kullanma. Akıcı sohbet yaz.

3. **UZUN VE DETAYLI YAZ:** En az 5-6 paragraf, doyurucu bir analiz yap. Kısa cevap verme!

4. **RAPORA BAĞLI KAL:** Raporda ne yazıyorsa onu söyle. Varsayım yapma, gördüğünü anlat.

5. **SOHBET GEÇMİŞİNİ HATIRLA:** Kullanıcı daha önce ne sorduysa, ona atıfta bulun.

Haydi Abla, bu verilere dayanarak kullanıcının sorusuna UZUN ve DETAYLI bir cevap ver!
"""

    return content


# ============================================
# ANA NODE FONKSİYONU
# ============================================
def persona_node(state: AgentState) -> Dict[str, Any]:
    """
    Toplanan tüm teknik verileri 'Abla' personasıyla kullanıcıya sunar.

    ÖNEMLİ: Bu node artık kullanıcının SORUSUNA özel cevap veriyor!
    Sadece el yorumu yapmıyor, soruyu el bulgularıyla ilişkilendiriyor.

    Args:
        state: Mevcut graph state'i (AgentState)

    Returns:
        Dict[str, Any]: State güncellemeleri
            - final_response: Abla'nın Türkçe yorumu
            - error_message: Hata varsa mesaj

    Flow:
        1. State'den vision_report, retrieved_documents ve messages al
        2. Kullanıcı sorusunu çıkar
        3. System prompt (Abla personası) hazırla
        4. User content (soru + teknik veri + referanslar) hazırla
        5. GPT-4o'ya gönder
        6. Türkçe yorumu state'e ekle
    """
    logger.info("--- 🗣️ ABLA NODE: Fal Yazılıyor... ---")

    # ==========================================
    # ADIM 1: Verileri Al
    # ==========================================
    vision_report = state.get("visual_analysis_report", "")
    book_references = state.get("retrieved_documents", [])
    messages = state.get("messages", [])  # Kullanıcı mesajları

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
    # ADIM 2: Kullanıcı Sorusunu ve Sohbet Geçmişini Çıkar
    # ==========================================
    user_question = _extract_user_question(messages)
    chat_history = _build_chat_history_text(messages)  # YENİ: Sohbet geçmişi

    logger.info(f"   🎯 Kullanıcı sorusu: '{user_question[:50]}...'")
    logger.info(f"   📜 Sohbet geçmişi: {len(messages)} mesaj")

    # ==========================================
    # ADIM 3: Modeli Hazırla
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
    # ADIM 4: Mesajları Hazırla
    # ==========================================
    # System message: Abla personası (güçlendirilmiş)
    system_message = SystemMessage(content=ABLA_SYSTEM_PROMPT)

    # User message: Sohbet geçmişi + Soru + Teknik veri + Referanslar
    user_content = _build_user_content(
        vision_report=vision_report,
        book_references=book_references,
        user_question=user_question,
        chat_history=chat_history  # YENİ: Sohbet geçmişi eklendi!
    )
    user_message = HumanMessage(content=user_content)

    messages_payload = [system_message, user_message]

    logger.debug(f"   📨 User content uzunluğu: {len(user_content)} karakter")

    # ==========================================
    # ADIM 5: API Çağrısı
    # ==========================================
    try:
        logger.info("   🔄 Abla düşünüyor...")
        response = llm.invoke(messages_payload)
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
    # ADIM 6: Sonucu Döndür
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