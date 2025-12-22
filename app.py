"""
============================================
YASAA VISION - Streamlit Chat Interface
============================================
Kullanıcıların fotoğraf yükleyip "Abla" ile
SOHBET edebileceği chat tabanlı arayüz.

Özellikler:
- Mistik/Karanlık tema
- Chat arayüzü (WhatsApp tarzı)
- Session State ile hafıza (konuşma geçmişi korunur)
- Fotoğraf bir kere yüklenir, sonra sohbet devam eder
- Akademik referans gösterimi

Çalıştırma:
    streamlit run app.py

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os                                      # Environment değişkenleri için
import base64                                  # Görsel encoding için
import logging                                 # Profesyonel loglama
from typing import Optional, Dict, Any, List   # Type hints için

import streamlit as st                         # Ana UI framework
from PIL import Image                          # Görsel işleme
from dotenv import load_dotenv                 # .env dosyası okuma
from langchain_core.messages import (          # Mesaj formatları
    HumanMessage,
    AIMessage
)

# Kendi modüllerimiz
from App.agent.graph import build_graph        # LangGraph akışı
from App.agent.state import AgentState         # State tipi


# ============================================
# ENVIRONMENT DEĞİŞKENLERİ
# ============================================
# .env dosyasını yükle
load_dotenv()

# --- UI Ayarları ---
APP_TITLE: str = os.getenv("APP_TITLE", "Yasaa Vision")
APP_SUBTITLE: str = os.getenv("APP_SUBTITLE", "Dijital Abla")
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"


# ============================================
# LOGGING AYARLARI
# ============================================
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================
# SAYFA KONFİGÜRASYONU
# ============================================
st.set_page_config(
    page_title=f"{APP_TITLE} - {APP_SUBTITLE}",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================
# ÖZEL CSS STİLLERİ (MİSTİK TEMA)
# ============================================
CUSTOM_CSS = """
<style>
/* Ana Arka Plan */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    color: #FAFAFA;
}

/* Başlık */
.main-title {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #9D4EDD, #E0AAFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: bold;
    margin-bottom: 0px;
}

/* Alt Başlık */
.subtitle {
    font-size: 1rem;
    color: #E0AAFF;
    text-align: center;
    margin-bottom: 20px;
    font-style: italic;
}

/* Chat Mesaj Kutuları */
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.chat-message.user {
    background-color: #2b313e;
    border-left: 3px solid #9D4EDD;
}

.chat-message.assistant {
    background-color: #1a1a2e;
    border-left: 3px solid #E0AAFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}

/* Buton */
.stButton > button {
    background: linear-gradient(90deg, #9D4EDD, #7B2CBF);
    color: white;
    border: none;
    border-radius: 10px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #7B2CBF, #5A189A);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================
# SESSION STATE BAŞLATMA (HAFIZA)
# ============================================
def initialize_session_state() -> None:
    """
    Streamlit session state'ini başlatır.

    Session state, sayfa yenilendiğinde bile verileri korur.
    Kullanıcı aynı oturumda kaldığı sürece:
    - Sohbet geçmişi korunur
    - Yüklenen fotoğraf hafızada kalır
    - El analiz raporu saklanır
    """
    # Sohbet geçmişi (HumanMessage ve AIMessage listesi)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("📝 Session state: messages oluşturuldu")

    # Yüklenen el fotoğrafı (Base64 formatında)
    if "uploaded_image_base64" not in st.session_state:
        st.session_state.uploaded_image_base64 = None
        logger.info("📸 Session state: uploaded_image_base64 oluşturuldu")

    # El analiz raporu (Vision Node'dan gelen)
    if "vision_report_memory" not in st.session_state:
        st.session_state.vision_report_memory = None
        logger.info("📋 Session state: vision_report_memory oluşturuldu")


# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================
def encode_image_to_base64(uploaded_file) -> Optional[str]:
    """
    Streamlit'in UploadedFile objesini Base64 string'e çevirir.

    Args:
        uploaded_file: Streamlit file uploader'dan gelen dosya

    Returns:
        Optional[str]: Base64 encoded string veya None
    """
    if uploaded_file is None:
        return None

    try:
        bytes_data = uploaded_file.getvalue()
        base64_string = base64.b64encode(bytes_data).decode("utf-8")
        logger.info(f"📸 Görsel encode edildi: {len(base64_string)} karakter")
        return base64_string
    except Exception as e:
        logger.error(f"❌ Görsel encode hatası: {e}")
        return None


def clear_chat_history() -> None:
    """
    Sohbet geçmişini temizler.
    Yeni bir konuşma başlatmak için kullanılır.
    """
    st.session_state.messages = []
    st.session_state.vision_report_memory = None
    logger.info("🗑️ Sohbet geçmişi temizlendi")


# ============================================
# SIDEBAR (YAN PANEL - FOTOĞRAF YÜKLEME)
# ============================================
def render_sidebar() -> None:
    """
    Sol taraftaki paneli oluşturur.
    - Fotoğraf yükleme alanı
    - Kullanım talimatları
    - Sohbet temizleme butonu
    """
    with st.sidebar:
        # Logo / Başlık
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <span style="font-size: 3rem;">🔮</span>
            <h2 style="color: #E0AAFF; margin: 0;">Yasaa Vision</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- Fotoğraf Yükleme ---
        st.markdown("### 📸 El Fotoğrafı")

        uploaded_file = st.file_uploader(
            "Avuç içi görünecek şekilde yükle",
            type=["jpg", "jpeg", "png", "webp"],
            help="Net, aydınlık bir el fotoğrafı seç"
        )

        # Fotoğraf yüklendiyse
        if uploaded_file is not None:
            # Görseli göster
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Senin Elin", use_container_width=True)
            except Exception as e:
                st.error(f"Görsel açılamadı: {e}")
                return

            # Base64'e çevir ve hafızaya al
            encoded_img = encode_image_to_base64(uploaded_file)

            # Yeni fotoğraf mı kontrol et
            if encoded_img != st.session_state.uploaded_image_base64:
                st.session_state.uploaded_image_base64 = encoded_img
                # Yeni fotoğraf = Yeni analiz gerekli
                st.session_state.vision_report_memory = None
                st.success("✅ Fotoğraf hafızaya alındı!")
                logger.info("📸 Yeni fotoğraf yüklendi")

        st.markdown("---")

        # --- Kullanım Talimatları ---
        st.markdown("### 📖 Nasıl Kullanılır?")
        st.info("""
        1. **Fotoğraf yükle** (bir kere yeterli)
        2. **Soru sor** - Abla cevaplasın
        3. **Tekrar sor** - Sohbet devam etsin
        
        Abla seni ve elini hatırlıyor! 🔮
        """)

        st.markdown("---")

        # --- Sohbeti Temizle ---
        if st.button("🗑️ Yeni Sohbet Başlat", use_container_width=True):
            clear_chat_history()
            st.rerun()

        # --- Debug Modu ---
        if DEBUG_MODE:
            st.markdown("---")
            st.error("🔧 DEBUG MODU AKTİF")
            st.caption(f"Mesaj sayısı: {len(st.session_state.messages)}")
            st.caption(f"Fotoğraf: {'Var' if st.session_state.uploaded_image_base64 else 'Yok'}")
            st.caption(f"Rapor: {'Var' if st.session_state.vision_report_memory else 'Yok'}")


# ============================================
# SOHBET GEÇMİŞİNİ GÖSTER
# ============================================
def render_chat_history() -> None:
    """
    Önceki mesajları ekrana basar.
    Her mesaj için uygun avatar ve stil kullanır.
    """
    for message in st.session_state.messages:
        # Kullanıcı mesajı
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

        # Abla'nın cevabı
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🔮"):
                st.markdown(message.content)


# ============================================
# FAL ANALİZ İŞLEMİ
# ============================================
def process_user_message(user_input: str) -> None:
    """
    Kullanıcının mesajını işler ve Abla'nın cevabını alır.

    Args:
        user_input: Kullanıcının yazdığı mesaj

    Bu fonksiyon:
    1. Mesajı sohbet geçmişine ekler
    2. Fotoğraf kontrolü yapar
    3. Graph'ı çalıştırır
    4. Cevabı gösterir ve hafızaya ekler
    """
    # --- 1. Kullanıcı mesajını ekrana bas ve hafızaya ekle ---
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    # --- 2. Fotoğraf kontrolü ---
    if not st.session_state.uploaded_image_base64:
        with st.chat_message("assistant", avatar="🔮"):
            error_msg = "Kuzum önce soldan bir el fotoğrafı yükle ki bakayım! 📸"
            st.warning(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
        return

    # --- 3. Abla düşünüyor ---
    with st.chat_message("assistant", avatar="🔮"):
        with st.spinner("🔮 Yıldızlara ve Benham'a bakıyorum... Sabret kuzum..."):
            try:
                # Graph'ı oluştur
                app = build_graph()
                logger.info("🧠 Graph oluşturuldu")

                # Input state hazırla
                # ÖNEMLİ: Tüm mesaj geçmişini gönder!
                inputs = {
                    "user_image_bytes": st.session_state.uploaded_image_base64,
                    "messages": st.session_state.messages,  # Tüm geçmiş
                    "visual_analysis_report": st.session_state.vision_report_memory,  # Önceki rapor (varsa)
                    "retrieved_documents": [],
                    "final_response": None,
                    "is_hand_detected": False,
                    "error_message": None
                }

                logger.info(f"📤 Input hazır: {len(st.session_state.messages)} mesaj")

                # Graph'ı çalıştır
                final_state = app.invoke(inputs)
                logger.info("✅ Graph çalıştırıldı")

                # Sonuçları al
                response_text = final_state.get("final_response")
                vision_report = final_state.get("visual_analysis_report")
                error_message = final_state.get("error_message")
                is_hand = final_state.get("is_hand_detected", False)

                # Vision raporunu hafızaya kaydet (bir sonraki soru için)
                if vision_report:
                    st.session_state.vision_report_memory = vision_report
                    logger.info("📋 Vision raporu hafızaya kaydedildi")

                # --- 4. Sonucu göster ---

                # Hata varsa
                if error_message:
                    st.error(f"🚫 {error_message}")
                    st.session_state.messages.append(AIMessage(content=error_message))
                    logger.warning(f"Hata: {error_message}")

                # El tespit edilemedi
                elif not is_hand:
                    warning_msg = "👀 Kuzum ben burada el göremedim. Başka bir fotoğraf dener misin?"
                    st.warning(warning_msg)
                    st.session_state.messages.append(AIMessage(content=warning_msg))
                    logger.warning("El tespit edilemedi")

                # Başarılı - Abla'nın cevabı
                elif response_text:
                    st.markdown(response_text)
                    st.session_state.messages.append(AIMessage(content=response_text))
                    logger.info(f"✅ Cevap alındı: {len(response_text)} karakter")

                    # Akademik referansları göster (opsiyonel)
                    retrieved_docs = final_state.get("retrieved_documents", [])
                    if retrieved_docs:
                        with st.expander("📚 Akademik Kaynaklar"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                preview = doc[:200] + "..." if len(doc) > 200 else doc
                                st.caption(f"**Referans {i}:** {preview}")

                else:
                    unknown_msg = "🤔 Bir şeyler yolunda gitmedi. Tekrar dener misin?"
                    st.warning(unknown_msg)
                    st.session_state.messages.append(AIMessage(content=unknown_msg))
                    logger.warning("Beklenmeyen durum: response_text boş")

            except Exception as e:
                error_msg = f"💥 Bir hata oluştu: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                logger.exception("İşlem hatası:")

                if DEBUG_MODE:
                    st.exception(e)


# ============================================
# ANA ARAYÜZ
# ============================================
def render_main_interface() -> None:
    """
    Ana sohbet arayüzünü oluşturur.
    - Başlık
    - Sohbet geçmişi
    - Mesaj girişi
    """
    # --- Başlık ---
    st.markdown(f'<p class="main-title">🔮 {APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">"Ben sadece gördüğümü söylerim, gerisi kader..." - Abla</p>', unsafe_allow_html=True)

    # --- Sohbet Geçmişi ---
    render_chat_history()

    # --- Mesaj Girişi ---
    user_input = st.chat_input("Abla'ya sorunu yaz...")

    if user_input:
        process_user_message(user_input)


# ============================================
# ANA UYGULAMA
# ============================================
def main() -> None:
    """
    Streamlit uygulamasının ana giriş noktası.
    """
    logger.info("🚀 Yasaa Vision UI başlatılıyor...")

    # Environment kontrolü
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY bulunamadı! `.env` dosyanızı kontrol edin.")
        st.stop()

    if not os.getenv("MONGO_URI"):
        st.error("❌ MONGO_URI bulunamadı! `.env` dosyanızı kontrol edin.")
        st.stop()

    # Session state başlat
    initialize_session_state()

    # Sidebar (fotoğraf yükleme)
    render_sidebar()

    # Ana arayüz (sohbet)
    render_main_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🔮 Yasaa Vision © 2024 | Kiromansi bilimini yapay zeka ile buluşturuyoruz
    </div>
    """, unsafe_allow_html=True)


# ============================================
# UYGULAMA GİRİŞ NOKTASI
# ============================================
if __name__ == "__main__":
    main()