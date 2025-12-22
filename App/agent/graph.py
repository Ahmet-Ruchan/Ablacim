"""
============================================
YASAA VISION - LangGraph Workflow (Beyin)
============================================
Bu dosya, tüm agent düğümlerini birleştirerek
ana iş akışını (workflow) oluşturur.

Akış:
    [Başlangıç]
         ↓
    [👁️ Gözcü] → El fotoğrafını analiz et
         ↓
    {El tespit edildi mi?}
         ↓ Evet          ↓ Hayır
    [📚 Araştırmacı]    [❌ Bitir]
         ↓
    [🗣️ Abla] → Fal yorumu yaz
         ↓
    [🏁 Bitiş]

Bu yapı bir DAG (Directed Acyclic Graph) oluşturur.
LangGraph bu graph'ı derler ve çalıştırılabilir hale getirir.
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import logging  # Profesyonel loglama
from typing import Literal  # Type hints için

from langgraph.graph import (  # LangGraph bileşenleri
    StateGraph,  # Graph oluşturucu
    END  # Bitiş node'u
)

# Kendi modüllerimiz
from app.agent.state import AgentState  # State tanımı
from app.agent.nodes.vision_node import vision_analysis_node  # Gözcü
from app.agent.nodes.retrieval_node import retrieval_node  # Araştırmacı
from app.agent.nodes.persona_node import persona_node  # Abla

# ============================================
# LOGGING AYARLARI
# ============================================
logger = logging.getLogger(__name__)


# ============================================
# ROUTER FONKSİYONLARI
# ============================================
def route_after_vision(state: AgentState) -> Literal["continue", "stop"]:
    """
    Gözcü'den sonra akışın nereye gideceğine karar verir.

    Bu fonksiyon bir "router" (yönlendirici) görevi görür.
    Gözcü'nün çıktısına bakarak:
    - El tespit edildiyse → Araştırmacı'ya git
    - El tespit edilmediyse → Akışı bitir

    Args:
        state: Mevcut graph state'i

    Returns:
        Literal["continue", "stop"]: Akış yönü

    LangGraph bu dönüş değerini conditional_edges'deki
    mapping ile eşleştirerek sonraki node'u belirler.
    """
    # State'den el tespit bilgisini al
    is_hand_detected = state.get("is_hand_detected", False)
    error_message = state.get("error_message")

    # Karar ver
    if is_hand_detected and not error_message:
        logger.info("   🚦 Router: El tespit edildi → Araştırmacı'ya git")
        return "continue"
    else:
        logger.warning("   🚦 Router: El tespit edilemedi veya hata var → Akışı bitir")
        return "stop"


# ============================================
# GRAPH BUILDER
# ============================================
def build_graph() -> StateGraph:
    """
    Yasaa Vision agent'ının iş akışını oluşturur ve derler.

    Bu fonksiyon:
    1. Boş bir StateGraph oluşturur
    2. Node'ları (düğümleri) ekler
    3. Edge'leri (bağlantıları) tanımlar
    4. Koşullu yönlendirmeleri ayarlar
    5. Graph'ı derler (compile)

    Returns:
        Compiled StateGraph: Çalıştırılmaya hazır graph

    Kullanım:
        >>> app = build_graph()
        >>> result = app.invoke(initial_state)

    veya streaming için:
        >>> for output in app.stream(initial_state):
        ...     print(output)
    """
    logger.info("🧠 Graph oluşturuluyor...")

    # ==========================================
    # ADIM 1: StateGraph Oluştur
    # ==========================================
    # AgentState tipinde bir graph başlat
    workflow = StateGraph(AgentState)

    logger.info("   📦 StateGraph oluşturuldu")

    # ==========================================
    # ADIM 2: Node'ları Ekle
    # ==========================================
    # Her node bir isim ve bir fonksiyon alır
    # Fonksiyon: state alır → güncellenmiş state parçası döndürür

    # 👁️ Gözcü: El fotoğrafını analiz eder
    workflow.add_node(
        "vision_scanner",  # Node adı (benzersiz)
        vision_analysis_node  # Çalıştırılacak fonksiyon
    )
    logger.info("   ✅ Node eklendi: vision_scanner (Gözcü)")

    # 📚 Araştırmacı: MongoDB'de arama yapar
    workflow.add_node(
        "knowledge_retriever",
        retrieval_node
    )
    logger.info("   ✅ Node eklendi: knowledge_retriever (Araştırmacı)")

    # 🗣️ Abla: Son yorumu üretir
    workflow.add_node(
        "fortune_teller",
        persona_node
    )
    logger.info("   ✅ Node eklendi: fortune_teller (Abla)")

    # ==========================================
    # ADIM 3: Başlangıç Noktasını Belirle
    # ==========================================
    # Graph'ın hangi node'dan başlayacağını söyle
    workflow.set_entry_point("vision_scanner")
    logger.info("   🚀 Başlangıç noktası: vision_scanner")

    # ==========================================
    # ADIM 4: Koşullu Yönlendirme (Conditional Edge)
    # ==========================================
    # Gözcü'den sonra: El varsa devam, yoksa bitir
    workflow.add_conditional_edges(
        "vision_scanner",  # Kaynak node
        route_after_vision,  # Router fonksiyonu
        {
            # Router'ın dönüş değeri → Hedef node
            "continue": "knowledge_retriever",  # El var → Araştırmacı
            "stop": END  # El yok → Bitir
        }
    )
    logger.info("   🔀 Koşullu edge eklendi: vision_scanner → (continue/stop)")

    # ==========================================
    # ADIM 5: Normal Edge'ler
    # ==========================================
    # Araştırmacı bittikten sonra → Abla'ya git
    workflow.add_edge("knowledge_retriever", "fortune_teller")
    logger.info("   ➡️ Edge eklendi: knowledge_retriever → fortune_teller")

    # Abla bittikten sonra → Akışı bitir
    workflow.add_edge("fortune_teller", END)
    logger.info("   ➡️ Edge eklendi: fortune_teller → END")

    # ==========================================
    # ADIM 6: Derleme (Compile)
    # ==========================================
    # Graph'ı çalıştırılabilir hale getir
    app = workflow.compile()
    logger.info("   ✅ Graph derlendi ve hazır!")

    return app


# ============================================
# GRAPH GÖRSELLEŞTİRME (OPSIYONEL)
# ============================================
def visualize_graph():
    """
    Graph yapısını görselleştirir (Mermaid formatında).

    Bu fonksiyon debug ve dokümantasyon için kullanışlıdır.
    Çıktıyı https://mermaid.live/ sitesinde görselleştirebilirsiniz.

    Returns:
        str: Mermaid formatında graph tanımı
    """
    mermaid_diagram = """
    ```mermaid
    graph TD
        Start((🚀 Başlangıç)) --> Vision[👁️ GÖZCÜ<br/>vision_scanner]

        Vision --> Router{El Tespit<br/>Edildi mi?}

        Router -- ✅ Evet --> Retriever[📚 ARAŞTIRMACI<br/>knowledge_retriever]
        Router -- ❌ Hayır --> ErrorEnd((❌ Hata<br/>Mesajı))

        Retriever --> Persona[🗣️ ABLA<br/>fortune_teller]

        Persona --> Success((🏁 Fal<br/>Tamamlandı))

        style Vision fill:#e1f5fe
        style Retriever fill:#fff3e0
        style Persona fill:#fce4ec
        style Router fill:#f3e5f5
    ```
    """
    return mermaid_diagram


# ============================================
# TEST FONKSİYONU
# ============================================
def _test_graph_build():
    """
    Graph'ın doğru oluşturulduğunu test eder.

    Kullanım:
        python -m App.agent.graph
    """
    print("=" * 50)
    print("🧠 Graph Build Test")
    print("=" * 50)

    print("\n📝 Test: Graph oluşturma")

    try:
        app = build_graph()
        print("   ✅ Graph başarıyla oluşturuldu!")

        # Graph bilgilerini göster
        print(f"\n📊 Graph Bilgileri:")
        print(f"   - Tip: {type(app)}")

        # Mermaid diagramını göster
        print("\n🎨 Graph Görselleştirme (Mermaid):")
        print(visualize_graph())

    except Exception as e:
        print(f"   ❌ Hata: {e}")

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
    _test_graph_build()