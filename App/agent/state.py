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

Yazar: Ahmet Ruçhan
Tarih: 2024
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
from typing import (
    TypedDict,      # Tip güvenli dictionary tanımı için
    List,           # Liste tipi için
    Optional,       # Opsiyonel (None olabilir) tipler için
    Annotated       # LangGraph için özel annotasyonlar
)
from langchain_core.messages import BaseMessage  # Chat mesaj tipi


# ============================================
# AGENT STATE TANIMI
# ============================================
class AgentState(TypedDict):
    """
    Graph boyunca taşınacak olan durum (hafıza) yapısı.

    Bu yapı bir "bavul" gibi düşünülebilir:
    - Her düğüm bavulu açar
    - İçindeki bilgileri okur
    - Kendi ürettiği bilgiyi ekler
    - Bavulu bir sonraki düğüme verir

    Attributes:
        messages: Kullanıcı ile olan chat geçmişi
        user_image_bytes: Kullanıcının gönderdiği el fotoğrafı (Base64)
        visual_analysis_report: Gözcü'nün teknik raporu
        retrieved_documents: MongoDB'den çekilen ilgili sayfalar
        final_response: Abla'nın son cevabı
        is_hand_detected: Fotoğrafın gerçekten el olup olmadığı
        error_message: Hata durumunda kullanıcıya gösterilecek mesaj
    """

    # ==========================================
    # 1. KULLANICIDAN GELENLER
    # ==========================================

    messages: List[BaseMessage]
    """
    Chat geçmişi - Kullanıcı ve asistanın önceki mesajları.
    LangChain'in BaseMessage formatında saklanır.
    Örnek: [HumanMessage("Elime bakar mısın?"), AIMessage("Tabii...")]
    """

    user_image_bytes: Optional[str]
    """
    Kullanıcının gönderdiği el fotoğrafı.
    Base64 formatında encode edilmiş string.
    
    Neden Base64?
    - Binary veriyi text olarak taşımak için
    - API'lere göndermek için standart format
    - JSON içinde taşınabilir
    
    None olabilir: Kullanıcı sadece soru soruyorsa resim olmayabilir.
    """

    # ==========================================
    # 2. GÖZCÜ'NÜN ÇIKTILARI (Vision Node)
    # ==========================================

    visual_analysis_report: Optional[str]
    """
    GPT-4o Vision'ın el hakkındaki teknik raporu.
    
    Örnek içerik:
    "Hand Shape: Square type based on equal palm width and finger length.
     Life Line: Deep and curved, extending around Mount of Venus.
     Head Line: Straight, ending near Mount of Moon.
     Heart Line: Curved, terminating under Mount of Jupiter.
     Prominent Mounts: Venus (padded), Jupiter (raised)."
    
    NOT: Bu rapor YORUM içermez, sadece gözlem içerir.
    Yorumu Abla yapacak.
    """

    # ==========================================
    # 3. ARAŞTIRMACI'NIN ÇIKTILARI (Retrieval Node)
    # ==========================================

    retrieved_documents: List[str]
    """
    MongoDB'den semantic search ile bulunan kitap sayfaları.
    
    Gözcü'nün raporundaki terimler (örn: "deep life line")
    kullanılarak veritabanında arama yapılır.
    
    Örnek:
    [
        "--- PAGE 322 ---\nThe deep Life line indicates...",
        "--- PAGE 145 ---\nWhen the Mount of Venus is padded..."
    ]
    
    Bu bilgiler Abla'ya "akademik kaynak" olarak verilir.
    """

    # ==========================================
    # 4. ABLA'NIN ÇIKTILARI (Persona Node)
    # ==========================================

    final_response: Optional[str]
    """
    Kullanıcıya dönecek son cevap.
    
    Abla persona'sı ile yazılmış, sıcak ve samimi ton.
    "Sandviç Tekniği" uygulanmış:
    - Olumlu giriş
    - Gerçekçi değerlendirme  
    - Motive edici kapanış
    
    Örnek:
    "Ay kuzum, ne güzel bir el! Şimdi bak sana ne diyeceğim...
     Hayat çizgin çok güçlü, demek ki dirençli birisin...
     Ama şurada dikkatli ol..."
    """

    # ==========================================
    # 5. KONTROL FLAG'LERİ
    # ==========================================

    is_hand_detected: bool
    """
    Gönderilen fotoğrafın gerçekten el olup olmadığı.
    
    True: Evet, bu bir el fotoğrafı → Analize devam et
    False: Hayır, el değil (kedi, manzara vb.) → Kullanıcıyı uyar
    
    Router düğümü bu flag'e bakarak akışı yönlendirir.
    """

    error_message: Optional[str]
    """
    Herhangi bir hata durumunda kullanıcıya gösterilecek mesaj.
    
    Olası durumlar:
    - Resim okunamadı
    - API hatası
    - El tespit edilemedi
    - Veritabanı bağlantı sorunu
    
    None ise: Her şey yolunda, hata yok.
    """


# ============================================
# STATE BAŞLATMA YARDIMCI FONKSİYONU
# ============================================
def create_initial_state(
    user_message: str = "",
    image_bytes: Optional[str] = None
) -> AgentState:
    """
    Yeni bir graph çalıştırması için başlangıç state'i oluşturur.

    Bu fonksiyon, boş bir "bavul" hazırlar.
    Graph başladığında ilk düğüm bu bavulu alır.

    Args:
        user_message: Kullanıcının ilk mesajı
        image_bytes: Varsa, Base64 formatında el fotoğrafı

    Returns:
        AgentState: Başlangıç değerleri ile doldurulmuş state

    Example:
        >>> state = create_initial_state(
        ...     user_message="Elime bakar mısın?",
        ...     image_bytes="base64_encoded_image_data..."
        ... )
    """
    from langchain_core.messages import HumanMessage

    # Başlangıç mesajını oluştur
    initial_messages = []
    if user_message:
        initial_messages.append(HumanMessage(content=user_message))

    # State'i döndür - tüm alanlar varsayılan değerlerle
    return AgentState(
        messages=initial_messages,
        user_image_bytes=image_bytes,
        visual_analysis_report=None,      # Henüz analiz yapılmadı
        retrieved_documents=[],            # Henüz arama yapılmadı
        final_response=None,               # Henüz cevap oluşturulmadı
        is_hand_detected=False,            # Henüz kontrol edilmedi
        error_message=None                 # Henüz hata yok
    )