import os
from dotenv import load_dotenv
from pymongo import MongoClient

# .env yükle (Ana dizinden)
# Not: Bu dosya App/ingest içinde olduğu için .env bir üst dizinin üstünde olabilir
# Garanti olsun diye path ayarı:
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Ablacim/
load_dotenv(os.path.join(project_root, ".env"))

# Ayarları Al
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "YasaaVisionDB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "palmistry_knowledge")


def clear_database():
    if not MONGO_URI:
        print("❌ HATA: .env dosyası okunamadı veya MONGO_URI eksik.")
        return

    print(f"🔌 MongoDB'ye bağlanılıyor... ({DB_NAME} / {COLLECTION_NAME})")

    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Mevcut kayıt sayısını say
        count_before = collection.count_documents({})
        print(f"📊 Mevcut Kayıt Sayısı: {count_before}")

        if count_before == 0:
            print("✅ Veritabanı zaten boş. Silinecek bir şey yok.")
            return

        # Kullanıcıdan son onay al (Güvenlik)
        confirm = input(f"⚠️ DİKKAT: {count_before} adet kayıt silinecek. Emin misin? (e/h): ")
        if confirm.lower() != 'e':
            print("❌ İşlem iptal edildi.")
            return

        # TÜM VERİYİ SİL (Index'i korur, sadece veriyi siler)
        result = collection.delete_many({})

        print(f"🗑️ SİLİNDİ: Toplam {result.deleted_count} belge yok edildi.")
        print("✨ Veritabanı tertemiz! Şimdi yeni 'Overlap'li ingestion işlemini yapabilirsin.")

    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")


if __name__ == "__main__":
    clear_database()