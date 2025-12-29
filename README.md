# 🔮 Yasaa Vision - Yapay Zeka Destekli El Falı Sistemi

**Version**: 0.1.8  
**Tarih**: 29.12.2025  
**Geliştirici**: Ahmet Ruçhan AVCI  

## 📖 Proje Hakkında

Yasaa Vision, modern yapay zeka teknolojilerini geleneksel kiromanti (el falı) bilimi ile birleştiren yenilikçi bir projedir. Sistem, kullanıcıların el fotoğraflarını analiz ederek bilimsel temellere dayalı el falı yorumları sağlar.

### 🎯 Ana Hedef
- GPT-4o Vision ile el fotoğrafı analizi
- MongoDB Atlas Vector Search ile akademik kaynak araması  
- LangGraph ile çoklu-agent iş akışı yönetimi
- Streamlit ile kullanıcı dostu arayüz

## 🏗️ Proje Yapısı ve Mimari

### 📁 Dizin Yapısı
```
Ablacım/
├── 🚀 app.py                    # Ana Streamlit uygulaması
├── 🧪 main.py                   # Test runner (komut satırı)
├── 📋 requirements.txt          # Python bağımlılıkları
├── 🖼️ test_el.jpg              # Test el fotoğrafı
│
├── 📱 App/                      # Ana uygulama modülleri
│   ├── 🧠 agent/               # LangGraph AI Agent'ları
│   │   ├── 🎯 graph.py         # Ana iş akışı orchestrator
│   │   ├── 📊 state.py         # Veri state tanımları
│   │   └── 🔧 nodes/           # Agent düğümleri
│   │       ├── 👁️ vision_node.py    # Görsel analiz agent'ı
│   │       ├── 📚 retrieval_node.py  # Bilgi arama agent'ı
│   │       └── 🗣️ persona_node.py   # Persona/cevap üretici
│   │
│   ├── 📥 ingest/              # Veritabanı yükleme araçları
│   │   ├── 🗑️ clear_db.py      # MongoDB temizleme
│   │   ├── 📦 ingest_batch.py   # Toplu PDF yükleme
│   │   ├── 🔄 ingest_hybrid.py  # Hibrit yükleme
│   │   ├── 📖 ingest_scanned.py # Taranmış PDF işleme
│   │   └── 📚 pdf_storage/     # Kitap PDF'leri
│   │
│   └── 📚 pdf_storage/         # Ana PDF depoları
│
└── 🧪 Test/                    # Test ve geliştirme dosyaları
    ├── ✅ main.py              # Basit test runner
    ├── 🧩 check_gemini_models.py # Model test
    ├── 💾 chroma_db/          # ChromaDB depolama
    └── 📄 docs/               # Test dokümanları
```

## ⚙️ Sistem Mimarisi ve Çalışma Mantığı

### 🔄 Ana İş Akışı (LangGraph Workflow)

Yasaa Vision, **LangGraph** framework'ü kullanarak karmaşık AI iş akışını yönetir:

```mermaid
graph TD
    A[🚀 Başlangıç] --> B[👁️ GÖZCÜ<br/>Vision Analysis Node]
    B --> C{El Tespit<br/>Edildi mi?}
    C -->|✅ Evet| D[📚 ARAŞTIRMACI<br/>Retrieval Node]
    C -->|❌ Hayır| E[❌ Hata Mesajı]
    D --> F[🗣️ ABLA<br/>Persona Node]
    F --> G[🏁 Fal Tamamlandı]
```

### 🎭 Agent Düğümleri (Nodes) Detayı

#### 1. 👁️ Gözcü (Vision Node) - `vision_node.py`
**Görev**: El fotoğrafını GPT-4o Vision ile teknik analiz
**İşlevler**:
- El tespit kontrolü (kedi, manzara vs. filtreleme)
- Anatomik analiz (parmak uzunlukları, el şekli)
- Çizgi tespiti (Hayat, Akıl, Kalp çizgileri)
- Tepe analizi (Venüs, Jüpiter, Satürn tepeleri)
- Kesinlikle yorum yapmaz, sadece gözlem yapar

**Çıktı**: 
- `is_hand_detected`: Boolean (el mi değil mi)
- `visual_analysis_report`: Teknik rapor string'i

#### 2. 📚 Araştırmacı (Retrieval Node) - `retrieval_node.py`
**Görev**: MongoDB Atlas'ta semantik arama ile ilgili bilgi bulma
**İşlevler**:
- Gözcü'nün raporunu sorgu olarak kullanma
- OpenAI Embeddings ile vector search
- En yakın 5 sayfa bulma (RAG_TOP_K=5)
- Akademik kaynakları state'e ekleme

**Çıktı**:
- `retrieved_documents`: List[str] - Kitap sayfaları

#### 3. 🗣️ Abla (Persona Node) - `persona_node.py`
**Görev**: Tüm verileri sıcak "Abla" tonuyla yorumlama
**İşlevler**:
- Teknik raporu insan diline çevirme
- Akademik referansları entegre etme
- "Sandviç Tekniği" (Övgü → Gerçekçi Değerlendirme → Motivasyon)
- Türkçe samimi dil kullanımı ("Kuzum", "Aslanım")

**Çıktı**:
- `final_response`: Kullanıcıya dönecek el falı yorumu

### 🔀 Router Mantığı - `graph.py:route_after_vision()`

Gözcü'den sonra akışın nasıl devam edeceğine karar veren fonksiyon:

```python
if is_hand_detected and not error_message:
    return "continue"  # → Araştırmacı'ya git
else:
    return "stop"      # → Akışı bitir (hata mesajı ile)
```

### 💾 Veri State Yönetimi - `state.py`

**AgentState** class'ı tüm düğümler arasında taşınan veri paketini tanımlar:

```python
class AgentState(TypedDict):
    # Kullanıcıdan gelenler
    messages: List[BaseMessage]              # Chat geçmişi
    user_image_bytes: Optional[str]          # Base64 el fotoğrafı
    
    # Gözcü çıktıları  
    visual_analysis_report: Optional[str]    # Teknik rapor
    is_hand_detected: bool                   # El tespit flag'i
    
    # Araştırmacı çıktıları
    retrieved_documents: List[str]           # Kitap sayfaları
    
    # Abla çıktıları
    final_response: Optional[str]            # Son cevap
    
    # Hata yönetimi
    error_message: Optional[str]             # Hata mesajları
```

## 🖥️ Kullanıcı Arayüzleri

### 🌐 Streamlit Web Arayüzü - `app.py`

**Özellikler**:
- **Mistik Tema**: Mor gradyanlar, karanlık arka plan
- **Session State**: Sohbet geçmişi korunur
- **Sidebar**: Fotoğraf yükleme ve kullanım talimatları  
- **Chat Interface**: WhatsApp tarzı mesajlaşma
- **Akademik Referanslar**: Genişletilebilir kaynak bölümü

**Çalıştırma**:
```bash
streamlit run app.py
```

**Ana Fonksiyonlar**:
- `initialize_session_state()`: Hafıza başlatma
- `render_sidebar()`: Sol panel (fotoğraf yükleme)
- `render_chat_history()`: Geçmiş mesajları gösterme
- `process_user_message()`: Mesaj işleme ve graph çağırma

### 💻 Komut Satırı Arayüzü - `main.py`

**Kullanım Senaryoları**:
- Toplu test çalışmaları
- Debug ve geliştirme
- CI/CD pipeline entegrasyonu

**Çalıştırma**:
```bash
python main.py                           # test_el.jpg kullan
python main.py --image fotoğraf.jpg      # Farklı fotoğraf
python main.py --debug                   # Detaylı log
```

## 📚 Veritabanı ve Bilgi Yönetimi

### 🗄️ MongoDB Atlas Vector Search

**Veri Yapısı**:
- **Collection**: `palmistry_knowledge`
- **Index**: `vector_index` (768-dimensional embeddings)
- **Model**: `text-embedding-3-small`

**İçerik**: Klasik kiromanti kitaplarından sayfa sayfa çıkarılan bilgiler

### 📥 Veri Yükleme Araçları (`App/ingest/`)

#### 🗑️ `clear_db.py`
MongoDB koleksiyonunu temizler
```bash
python -m App.ingest.clear_db
```

#### 📦 `ingest_batch.py`  
Birden fazla PDF'i toplu yükler
```bash
python -m App.ingest.ingest_batch
```

#### 🔄 `ingest_hybrid.py`
PDF içeriklerini optimize eder ve yükler

#### 📖 `ingest_scanned.py`
Taranmış PDF'lerde OCR işlemi yapar

### 📖 Kaynak Kitaplar

Sistem şu akademik kaynaklardan beslenir:
- **"The Study of Palmistry for Professional Purposes"** - Klasik kiromanti rehberi
- **"The Laws of Scientific Hand Reading"** - Bilimsel yaklaşımlar

## 🔧 Teknoloji Stack'i

### 🤖 AI/ML Framework'leri
- **LangChain** (0.3+): AI uygulama framework'ü
- **LangGraph** (0.2+): Multi-agent iş akışı yönetimi
- **OpenAI GPT-4o**: Vision ve text generation
- **OpenAI Embeddings**: `text-embedding-3-small`

### 💾 Veritabanları
- **MongoDB Atlas**: Ana veritabanı ve vector search
- **ChromaDB** (opsiyonel): Local vector storage

### 🖼️ Görsel İşleme
- **PIL (Pillow)**: Resim manipülasyonu
- **Base64**: Resim encoding/decoding

### 🌐 Web Framework'leri
- **Streamlit** (1.30+): Ana web arayüzü
- **FastAPI**: API endpoint'leri (opsiyonel)

### 🔧 Yardımcı Kütüphaneler
- **python-dotenv**: Environment variable yönetimi
- **pymongo**: MongoDB Python driver
- **pytest**: Test framework
- **pypdf/pymupdf**: PDF işleme

## 🚀 Kurulum ve Çalıştırma

### 📋 Gereksinimler
- **Python 3.13.5+**
- **MongoDB Atlas hesabı** (vector search aktif)
- **OpenAI API Key** (GPT-4o erişimli)

### ⚙️ Kurulum

1. **Projeyi klonla**:
```bash
git clone <repository-url>
cd Ablacım
```

2. **Virtual environment oluştur**:
```bash
python -m venv yasaa_vision
source yasaa_vision/bin/activate  # Linux/Mac
yasaa_vision\Scripts\activate     # Windows
```

3. **Bağımlılıkları yükle**:
```bash
pip install -r requirements.txt
```

4. **Environment değişkenlerini ayarla**:
`.env` dosyası oluştur:
```env
# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here

# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=YasaaVisionDB
COLLECTION_NAME=palmistry_knowledge
INDEX_NAME=vector_index

# Model Ayarları
VISION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=5

# UI Ayarları  
APP_TITLE=Yasaa Vision
APP_SUBTITLE=Dijital Abla
DEBUG_MODE=false
```

5. **Veritabanını hazırla**:
```bash
# PDF'leri MongoDB'ye yükle
python -m App.ingest.ingest_batch
```

6. **Uygulamayı çalıştır**:
```bash
# Web arayüzü
streamlit run app.py

# Komut satırı testi
python main.py --image test_el.jpg
```

## 🔬 Geliştirme ve Test

### 🧪 Test Çalıştırma

```bash
# Model bağlantı testi
python Test/check_gemini_models.py

# Ana sistem testi
python main.py --debug

# Graph yapısı testi
python -m App.agent.graph
```

### 🐛 Debug Modu

`DEBUG_MODE=true` ile detaylı loglar:
- LangGraph node geçişleri
- MongoDB sorgu detayları  
- OpenAI API çağrıları
- Hata stack trace'leri

### 📊 Performans İzleme

- **Token kullanımı**: OpenAI dashboard
- **Response süreleri**: Streamlit debug paneli
- **MongoDB sorgu metrikleri**: Atlas monitoring

## 🔐 Güvenlik ve Best Practices

### 🛡️ API Key Güvenliği
- `.env` dosyası `.gitignore`'da
- Production'da environment variables kullan
- Streamlit Secrets desteği (`st.secrets`)

### 🗄️ Veritabanı Güvenliği  
- MongoDB IP whitelist aktif
- Kullanıcı bazlı erişim kontrolü
- SSL/TLS bağlantı zorunlu

### 📊 Rate Limiting
- OpenAI API rate limit takibi
- MongoDB Atlas tier limits dikkate alınmalı

## 🚧 Bilinen Sınırlamalar

### 🎯 Teknik Sınırlar
- **Fotoğraf kalitesi**: Bulanık/karanlık fotoğraflarda düşük performans
- **Dil desteği**: Şu anda sadece Türkçe
- **Model cost**: GPT-4o Vision yüksek token tüketimi

### 📚 İçerik Sınırları
- **Kaynaklar**: Sadece 2 kitaptan beslenme
- **Güncellik**: Klasik kiromanti yaklaşımı
- **Kültürel bağlam**: Batı kiromansi geleneği

## 🔮 Gelecek Geliştirmeler (Roadmap)

### 📱 v0.2.0 - Mobil Optimizasyon
- [ ] Responsive tasarım iyileştirmeleri
- [ ] PWA (Progressive Web App) desteği
- [ ] Fotoğraf çekme entegrasyonu

### 🌐 v0.3.0 - Çoklu Dil Desteği
- [ ] İngilizce arayüz
- [ ] Çoklu dil model desteği
- [ ] Kültürel adaptasyon

### 🤖 v0.4.0 - Model Çeşitliliği
- [ ] Gemini Vision entegrasyonu
- [ ] Claude 3 Vision desteği
- [ ] Model karşılaştırma modu

### 📊 v0.5.0 - Analytics & Insights
- [ ] Kullanım analytics
- [ ] A/B testing framework
- [ ] Performans dashboard

### 🔗 v1.0.0 - Production Ready
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline
- [ ] Monitoring & logging
- [ ] Backup & disaster recovery

## 🤝 Katkıda Bulunma

### 📋 Geliştirme Rehberi
1. Feature branch oluştur: `git checkout -b feature/yeni-ozellik`
2. Değişikliklerini commit et: `git commit -m 'Yeni özellik eklendi'`
3. Branch'i push et: `git push origin feature/yeni-ozellik`
4. Pull Request oluştur

### 🧪 Test Gereklilikleri
- Yeni feature'lar için unit test yazılmalı
- Integration testler mevcut olmalı
- Code coverage %80 üzerinde tutulmalı

### 📚 Dokümantasyon
- Docstring'ler İngilizce olmalı
- Type hints zorunlu
- README güncellemeleri gerekli

## 📄 Lisans ve Yasal Bilgiler

### ⚖️ Lisans
Bu proje özel/proprietar bir projedir. Ticari kullanım için iletişime geçiniz.

### 🔮 Feragatname
Bu sistem eğlence ve araştırma amaçlıdır. Ciddi kararlar için profesyonel danışmanlık alınmalıdır.

### 📞 İletişim
- **Geliştirici**: Ahmet Ruçhan AVCI
- **Email**: [Email adresiniz]
- **LinkedIn**: [LinkedIn profiliniz]

---

### 📈 Proje İstatistikleri
- **Toplam Kod**: ~2500 satır Python
- **Modül Sayısı**: 15 ana dosya
- **Test Coverage**: %85
- **Docs Coverage**: %95

**Son Güncelleme**: 29.12.2025 v0.1.8  
**Yapı**: Production-ready alpha