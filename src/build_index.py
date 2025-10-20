"""
İndeksleme betiği:
- Hugging Face veri setini (Türk hukuk Soru-Cevap) indirir.
- Soru-Cevap çiftlerini tek bir metne çevirip küçük parçalara böler.
- Sentence-Transformers ile gömme vektörlerini hesaplayıp FAISS indeksine yazar.
- Her parça için meta.jsonl'de ham metni saklar; retrieval sırasında bağlam döndürülür.
Komut: `python -m src.build_index`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

from .config import INDEX_DIR, EMBEDDING_MODEL_NAME
from .utils import chunk_text, batched


DATASET_NAME = "Renicames/turkish-law-chatbot"
MAX_CHARS = 512
BATCH_SIZE = 256


def _row_to_text(row: dict) -> str:
    # Yalnızca cevap metnini indekse koy (UI'da Soru/Cevap kalıpları görünmesin)
    a = row.get("Cevap") or row.get("answer") or ""
    return a.strip()


def load_texts(limit: int | None = None) -> List[str]:
    ds = load_dataset(DATASET_NAME)
    texts: List[str] = []
    # Varsayılan tek split ya da train split olabilir
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    if not splits:
        splits = ["train"]
    for split in splits:
        for row in ds[split]:
            text = _row_to_text(row)
            for part in chunk_text(text, max_chars=MAX_CHARS):
                if part:
                    texts.append(part)
                    if limit and len(texts) >= limit:
                        return texts
    return texts


def main() -> None:
    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "meta.jsonl"

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Her çalıştırmada güncel veri setine göre yeniden oluştur
    if index_path.exists():
        try:
            index_path.unlink()
        except Exception:
            pass
    if meta_path.exists():
        try:
            meta_path.unlink()
        except Exception:
            pass

    print("[info] Loading dataset and preparing texts ...")
    texts = load_texts()
    print(f"[info] Total chunks: {len(texts)}")

    print(f"[info] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    index = faiss.IndexFlatIP(dim)
    normed_vectors = []

    with meta_path.open("w", encoding="utf-8") as fout:
        for batch in tqdm(batched(texts, BATCH_SIZE), total=(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE):
            embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            index.add(embeddings.astype(np.float32))
            for chunk in batch:
                fout.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(index_path))
    print(f"[ok] Index written to {index_path}")
    print(f"[ok] Metadata written to {meta_path}")


if __name__ == "__main__":
    main()



