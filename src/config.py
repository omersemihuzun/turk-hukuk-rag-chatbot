"""
Proje genel yapılandırmaları:
- Proje ve depolama yollarını belirler.
- Ortam değişkenlerinden gömme/LLM ayarlarını okur.
Bu dosya başka modüllerce import edilerek tek kaynaktan ayar sağlanır.
"""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = PROJECT_ROOT / "storage"
INDEX_DIR = STORAGE_DIR / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str | None:
    """Güvenli ortam değişkeni okuma yardımcı fonksiyonu."""
    value = os.getenv(name, default)
    return value


EMBEDDING_MODEL_NAME = get_env(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

LLM_PROVIDER = get_env("LLM_PROVIDER", "gemini")  # or "openai"
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")



