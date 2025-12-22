"""
============================================
YASAA VISION - Main Test Runner
============================================
Bu dosya, tüm sistemi uçtan uca test etmek için kullanılır.

Kullanım:
1. Proje kök dizinine bir el fotoğrafı koy (test_el.jpg)
2. Bu scripti çalıştır: python main.py
3. Abla'nın yorumunu gör!

Alternatif:
- --image argümanı ile farklı dosya belirt
- --debug argümanı ile detaylı log aç
============================================
"""

# ============================================
# IMPORTS - Gerekli Kütüphaneler
# ============================================
import os  # Dosya işlemleri için
import sys  # Sistem argümanları için
import base64  # Görsel encoding için
import logging  # Profesyonel loglama
import argparse  # Komut satırı argümanları
from typing import Optional  # Type hints için

from dotenv import load_dotenv  # .env dosyası okuma

# Kendi modüllerimiz
from App.agent.graph import build_graph  # Ana graph
from App.agent.state import AgentState  # State tipi


# ============================================
# LOGGING AYARLARI
# ============================================
def setup_logging(debug: bool = False) -> None:
    """
    Logging seviyesini ayarlar.

    Args:
        debug: True ise DEBUG seviyesi, False ise INFO
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ============================================
# GÖRSEL OKUMA VE ENCODE
# ============================================
def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Bir görsel dosyasını Base64 formatına çevirir.

    Args:
        image_path: Görsel dosyasının yolu

    Returns:
        Optional[str]: Base64 encoded string veya hata durumunda None

    Supported formats:
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - WebP (.webp)
    """
    # Dosya varlık kontrolü
    if not os.path.exists(image_path):
        logging.error(f"❌ Dosya bulunamadı: {image_path}")
        return None

    # Dosya boyutu kontrolü (çok büyük dosyalar API limitini aşabilir)
    file_size = os.path.getsize(image_path)
    max_size = 20 * 1024 * 1024  # 20 MB

    if file_size > max_size:
        logging.error(f"❌ Dosya çok büyük: {file_size / 1024 / 1024:.1f} MB (max: 20 MB)")
        return None

    # Dosyayı oku ve encode et
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_string = base64.b64encode(image_bytes).decode("utf-8")

        logging.info(f"   📸 Görsel yüklendi: {image_path} ({file_size / 1024:.1f} KB)")
        return base64_string

    except Exception as e:
        logging.error(f"❌ Görsel okuma hatası: {e}")
        return None


# ============================================
# BAŞLANGIÇ STATE OLUŞTURMA
# ============================================
def create_input_state(image_base64: str) -> AgentState:
    """
    Graph için başlangıç state'i oluşturur.

    Args:
        image_base64: Base64 encoded görsel verisi

    Returns:
        AgentState: Başlangıç state'i
    """
    return {
        "messages": [],  # Boş chat geçmişi
        "user_image_bytes": image_base64,  # Kullanıcının el fotoğrafı
        "visual_analysis_report": None,  # Henüz analiz yok
        "retrieved_documents": [],  # Henüz arama yok
        "final_response": None,  # Henüz cevap yok
        "is_hand_detected": False,  # Henüz kontrol edilmedi
        "error_message": None  # Henüz hata yok
    }


# ============================================
# ANA ÇALIŞTIRMA FONKSİYONU
# ============================================
def run_fortune_telling(image_path: str) -> None:
    """
    El falı sürecini baştan sona çalıştırır.

    Args:
        image_path: El fotoğrafının yolu

    Bu fonksiyon:
    1. Görseli yükler ve encode eder
    2. Graph'ı oluşturur
    3. Başlangıç state'i hazırlar
    4. Graph'ı çalıştırır (streaming mode)
    5. Sonucu ekrana yazdırır
    """
    print("\n" + "=" * 60)
    print("🔮 YASAA VISION - El Falı Sistemi")
    print("=" * 60)

    # ==========================================
    # ADIM 1: Görseli Yükle
    # ==========================================
    print("\n📸 [1/4] Görsel yükleniyor...")

    image_base64 = encode_image_to_base64(image_path)

    if not image_base64:
        print("\n❌ Görsel yüklenemedi. Lütfen geçerli bir dosya yolu verin.")
        return

    # ==========================================
    # ADIM 2: Graph'ı Oluştur
    # ==========================================
    print("\n🧠 [2/4] Yapay zeka hazırlanıyor...")

    try:
        app = build_graph()
        print("   ✅ Graph oluşturuldu")
    except Exception as e:
        print(f"\n❌ Graph oluşturma hatası: {e}")
        return

    # ==========================================
    # ADIM 3: Başlangıç State'i Hazırla
    # ==========================================
    print("\n🎯 [3/4] Analiz başlatılıyor...")

    input_state = create_input_state(image_base64)

    # ==========================================
    # ADIM 4: Graph'ı Çalıştır (Streaming)
    # ==========================================
    print("\n🌊 [4/4] Akış başlıyor...\n")
    print("-" * 40)

    final_output = None
    error_occurred = False

    try:
        # Streaming mode - her node tamamlandığında çıktı al
        for output in app.stream(input_state):
            # Her node'un çıktısını işle
            for node_name, node_output in output.items():
                print(f"   📍 {node_name} tamamlandı")

                # Son node'un çıktısını sakla
                final_output = node_output

                # Hata kontrolü
                if node_output.get("error_message"):
                    error_occurred = True
                    break

            if error_occurred:
                break

    except Exception as e:
        print(f"\n❌ Çalıştırma hatası: {e}")
        logging.exception("Detaylı hata:")
        return

    print("-" * 40)

    # ==========================================
    # ADIM 5: Sonucu Göster
    # ==========================================
    print("\n" + "=" * 60)

    # Hata durumu
    if error_occurred and final_output:
        error_msg = final_output.get("error_message", "Bilinmeyen hata")
        print("❌ HATA:")
        print("=" * 60)
        print(error_msg)

    # Başarılı durum - Abla'nın cevabı
    elif final_output and final_output.get("final_response"):
        print("🔮 ABLA'NIN YORUMU:")
        print("=" * 60)
        print(final_output["final_response"])

    # El tespit edilemedi
    elif final_output and not final_output.get("is_hand_detected", True):
        print("⚠️ EL TESPİT EDİLEMEDİ:")
        print("=" * 60)
        print(final_output.get("error_message", "Lütfen net bir el fotoğrafı gönderin."))

    # Beklenmeyen durum
    else:
        print("⚠️ BEKLENMEDİK DURUM:")
        print("=" * 60)
        print("Bir şeyler yolunda gitmedi. Lütfen tekrar deneyin.")

    print("\n" + "=" * 60)


# ============================================
# KOMUT SATIRI ARGÜMANLARI
# ============================================
def parse_arguments():
    """
    Komut satırı argümanlarını parse eder.

    Returns:
        argparse.Namespace: Parse edilmiş argümanlar
    """
    parser = argparse.ArgumentParser(
        description="🔮 Yasaa Vision - El Falı Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py                      # Varsayılan test_el.jpg kullan
  python main.py --image el_fotom.png # Farklı dosya kullan
  python main.py --debug              # Detaylı log aç
        """
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default="test_el.jpg",
        help="El fotoğrafının yolu (varsayılan: test_el.jpg)"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Debug modunu aktif et (detaylı log)"
    )

    return parser.parse_args()


# ============================================
# ANA GİRİŞ NOKTASI
# ============================================
def main():
    """
    Ana giriş noktası.

    Environment değişkenlerini yükler ve el falı sürecini başlatır.
    """
    # Argümanları parse et
    args = parse_arguments()

    # Logging'i ayarla
    setup_logging(debug=args.debug)

    # .env dosyasını yükle
    load_dotenv()

    # API key kontrolü
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ HATA: OPENAI_API_KEY bulunamadı!")
        print("   Lütfen .env dosyanızı kontrol edin.")
        sys.exit(1)

    if not os.getenv("MONGO_URI"):
        print("\n❌ HATA: MONGO_URI bulunamadı!")
        print("   Lütfen .env dosyanızı kontrol edin.")
        sys.exit(1)

    # Görsel dosyası kontrolü
    if not os.path.exists(args.image):
        print(f"\n❌ HATA: '{args.image}' bulunamadı!")
        print("\n💡 Çözüm önerileri:")
        print(f"   1. '{args.image}' dosyasını proje dizinine koyun")
        print("   2. --image argümanı ile farklı dosya belirtin:")
        print(f"      python main.py --image /path/to/your/image.jpg")
        print("\n   İnternetten bir el fotoğrafı indirebilir veya")
        print("   kendi elinizin fotoğrafını çekebilirsiniz.")
        sys.exit(1)

    # El falını başlat!
    run_fortune_telling(args.image)


# ============================================
# MODÜL DOĞRUDAN ÇALIŞTIRILIRSA
# ============================================
if __name__ == "__main__":
    main()