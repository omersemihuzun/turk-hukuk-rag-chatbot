## Türk Hukuk RAG Chatbot

Bu proje, `Renicames/turkish-law-chatbot` hukuk Soru-Cevap veri seti üzerinde RAG (Retrieval Augmented Generation) mimarisiyle bir hukuk danışmanı sohbet botu geliştirir ve Streamlit tabanlı bir web arayüzü üzerinden sunar.

### Amaç
- Kullanıcının Türkçe hukuki sorularına, veri setinden ilgili bağlam parçalarını geri getirip büyük dil modeli ile zenginleştirilmiş yanıtlar üretmek.
- Konuşma geçmişini analiz ederek bağlamsal devamlılık sağlamak.

- <img width="1918" height="995" alt="Ekran görüntüsü 2025-10-22 204614" src="https://github.com/user-attachments/assets/880e791c-0aa8-44eb-9ab7-ffdd95aff851" />

<img width="1918" height="985" alt="Ekran görüntüsü 2025-10-22 204654" src="https://github.com/user-attachments/assets/ce34edc7-8aa9-4b4d-8486-4976c491eb35" />

### Veri Seti Hakkında
**Veri Seti:** `Renicames/turkish-law-chatbot`

**İçerik:**
- **13,354** Türkçe hukuk Soru-Cevap çifti
- **Train:** 13,354 kayıt, **Test:** 1,500 kayıt
- Anayasa, kanunlar, mevzuat ve temel hukuk kavramları
- Türk hukuk sistemi, temel haklar, devlet yapısı konuları

**Veri Yapısı:**
- `Soru`: Hukuki soru metni
- `Cevap`: Detaylı hukuki açıklama

**Kullanım Amacı:**
Bu projede hukuki sorular girildiğinde ilgili bağlamlar bulunup LLM ile yanıt zenginleştirilir. RAG sistemi sayesinde veri setindeki bilgilerden yararlanarak doğru ve bağlamsal hukuki yanıtlar üretilir.

### Kullanılan Yöntemler ve Mimariler
- **RAG**: FAISS tabanlı yerel vektör indeksi + gömleme modeli + LLM üretimi.
- **Gömleme (embeddings)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (çok dilli, Türkçe uyumlu).
- **Vektör veritabanı**: FAISS (yerel disk üzerinde kalıcı).
- **Yeniden sıralama**: CrossEncoder ile daha alakalı bağlam parçalarını öne çıkarma.
- **Konuşma geçmişi**: Son mesajlardan anahtar kelimeleri çıkarıp bağlamsal devamlılık sağlama.
- **Üretim (LLM)**: Google Gemini (Gemini API) veya OpenAI API; ortam değişkeni ile seçilebilir.

### Elde Edilen Sonuçlar (Özet)
- İlgili bağlam parçaları geri getirildiğinde, LLM yanıtlarının doğruluğu ve bağlam tutarlılığı artar.
- Konuşma geçmişi analizi sayesinde "örnek ver" gibi belirsiz sorulara daha alakalı yanıtlar üretilir.
- Her yanıtta kaynak madde bilgisi otomatik olarak eklenir.

### Proje Yapısı
```
.
├─ app.py                     # Streamlit arayüzü (sohbet UI)
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ config.py               # Ortam değişkenleri ve yol ayarları
│  ├─ build_index.py          # HF dataset -> temizleme -> gömme -> FAISS indeks
│  ├─ rag_pipeline.py         # Retriever, CrossEncoder ve LLM zinciri
│  └─ utils.py                # Yardımcı fonksiyonlar (chunking, batching)
└─ notebooks/
   └─ RAG_Hukuk.ipynb   # Açıklamalı notebook



### Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Ortam Değişkenleri
**Güvenli yöntem (önerilen):**
1. `.env.example` dosyasını `.env` olarak kopyalayın
2. `.env` dosyasındaki API anahtarlarınızı doldurun
3. `.env` dosyası `.gitignore`'da olduğu için GitHub'a yüklenmez

**Manuel yöntem:**
- `EMBEDDING_MODEL` (opsiyonel, varsayılan: sentence-transformers çok dilli MiniLM)
- `GEMINI_API_KEY` (opsiyonel, Gemini kullanacaksanız)
- `OPENAI_API_KEY` (opsiyonel, OpenAI kullanacaksanız)
- `LLM_PROVIDER` (opsiyonel: `gemini` | `openai`, varsayılan: `gemini`)

Windows PowerShell örneği:
```powershell
$env:GEMINI_API_KEY = "YOUR_KEY"
$env:LLM_PROVIDER = "gemini"
```

### İndeks Oluşturma
```bash
python -m src.build_index
```
Bu komut, Hugging Face üzerinden veri setini indirir, metinleri parçalara böler, gömmeleri hesaplar ve `./storage/faiss_index` dizinine FAISS indeksini yazar.

### Web Arayüzünü Çalıştırma
```bash
streamlit run app.py
```
Tarayıcıda açılan arayüzde Türkçe hukuki sorular sorabilir, RAG ile üretilen yanıtları görebilirsiniz. Sohbet geçmişi otomatik olarak tutulur.

### Özellikler
- **Konuşma geçmişi**: Son mesajlardan anahtar kelimeleri çıkararak bağlamsal devamlılık
- **Yeniden sıralama**: CrossEncoder ile daha alakalı bağlam parçalarını öne çıkarma
- **Kaynak gösterimi**: Her yanıtta otomatik madde numarası ipucu
- **Fallback**: LLM anahtarı yoksa yerel özet üretimi

### Deploy
- Yerel çalıştırma için yukarıdaki adımlar yeterlidir.
- İsteğe bağlı olarak Streamlit Community Cloud, Hugging Face Spaces (Gradio/Streamlit) veya Docker + herhangi bir PaaS üzerinde dağıtabilirsiniz.
- [Web Uygulaması] https://turk-hukuk-rag-chatbot.streamlit.app/

### Kaynaklar
- Gemini API Docs: [ai.google.dev](https://ai.google.dev/gemini-api/docs)
- Gemini API Cookbook: [ai.google.dev/cookbook](https://ai.google.dev/gemini-api/cookbook)
- Haystack: [haystack.deepset.ai](https://haystack.deepset.ai/)
- Chatbot Templates: [github.com/enesmanan/chatbot-deploy](https://github.com/enesmanan/chatbot-deploy)
- LLM Toolkit: [github.com/KalyanKS-NLP/llm-engineer-toolkit](https://github.com/KalyanKS-NLP/llm-engineer-toolkit)
- Türk Hukuk veri seti: [Hugging Face](https://huggingface.co/datasets/Renicames/turkish-law-chatbot)

### Lisans
- Veri seti: Hugging Face veri seti sayfasına bakınız.
