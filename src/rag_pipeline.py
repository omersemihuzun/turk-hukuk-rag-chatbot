"""
RAG boru hattı:
- FAISS'ten benzer bağlamları getirir (retriever).
- CrossEncoder ile yeniden sıralar (daha alakalı ilk parçalar).
- Seçili LLM'e (Gemini/OpenAI) bağlam + soru verip yanıt üretir; anahtar yoksa yerel fallback.
- Basit bir madde numarası çıkarımı ile "Kaynak: Anayasa madde X" ipucu üretir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import re

from .config import INDEX_DIR, EMBEDDING_MODEL_NAME, LLM_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY


class FaissRetriever:
    def __init__(self, index_path: Path, meta_path: Path, embedding_model_name: str) -> None:
        self.index = faiss.read_index(str(index_path))
        self.metadata: List[str] = []
        with meta_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                self.metadata.append(obj["text"])  # aligned by add order
        self.model = SentenceTransformer(embedding_model_name, device="cpu")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # Daha geniş aday havuzu al ve sonra yeniden sırala
        base_k = max(20, top_k * 4)
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = q.astype(np.float32)
        scores, idxs = self.index.search(q, base_k)
        candidates: List[Tuple[str, float]] = []
        for i, score in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            candidates.append((self.metadata[i], float(score)))

        # CrossEncoder ile rerank (çok dilli küçük model)
        try:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            pairs = [(query, ctx) for ctx, _ in candidates]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(zip(candidates, rerank_scores), key=lambda x: float(x[1]), reverse=True)
            results = [(ctx, float(score)) for (ctx, _), score in ranked[:top_k]]
            return results
        except Exception:
            # Reranker indirilemezse FAISS skoruna göre devam et
            return candidates[:top_k]


def _init_gemini():
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


def _init_openai():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY tanımlı değil.")
    return OpenAI()


def _fallback_answer(question: str, contexts: List[Tuple[str, float]] | None) -> str:
    if not contexts:
        return "Bilmiyorum."
    # En iyi 2 parçayı kısa bir özet olarak döndür
    parts = [ctx for ctx, _ in contexts[:2]]
    joined = "\n\n".join(parts)
    if len(joined) > 800:
        joined = joined[:800] + "..."
    return f"(Yerel özet) Soruya ilişkin bağlamdan çıkarım: \n{joined}"


def generate_answer(prompt: str, question: str, contexts: List[Tuple[str, float]] | None) -> str:
    provider = (LLM_PROVIDER or "gemini").lower()
    try:
        if provider == "gemini":
            if not GEMINI_API_KEY:
                return _fallback_answer(question, contexts)
            model = _init_gemini()
            resp = model.generate_content(prompt)
            return resp.text or _fallback_answer(question, contexts)
        if provider == "openai":
            if not OPENAI_API_KEY:
                return _fallback_answer(question, contexts)
            client = _init_openai()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        return _fallback_answer(question, contexts)
    except Exception:
        return _fallback_answer(question, contexts)


def format_prompt(question: str, contexts: List[Tuple[str, float]], chat_history: List[dict] = None) -> str:
    context_block = "\n\n".join([f"[Parça {i+1} | benzerlik={score:.3f}]\n{ctx}" for i, (ctx, score) in enumerate(contexts)])
    
    # Konuşma geçmişinden anahtar kelimeleri çıkar
    history_context = ""
    if chat_history and len(chat_history) > 1:
        recent_topics = []
        for msg in chat_history[-3:]:  # Son 3 mesaj
            if msg["role"] == "assistant":
                # Yanıtlardan anahtar kelimeleri çıkar (basit regex)
                text = msg["content"].lower()
                keywords = re.findall(r'\b(anayasa|madde|kanun|devrim|türkiye|cumhuriyet|hukuk|yasa|mevzuat)\b', text)
                recent_topics.extend(keywords)
        
        if recent_topics:
            unique_topics = list(set(recent_topics))[:5]  # En fazla 5 benzersiz konu
            history_context = f"\n\nÖnceki konuşma konuları: {', '.join(unique_topics)}"
    
    system = (
        "Rolün bir Türk hukuk danışmanı. Cevapları Türkçe, kısa ve maddi hatadan kaçınarak ver. "
        "Sadece BAĞLAM içindeki bilgiye dayan; dış bilgi ekleme. Mevzuat/madde numarası BAĞLAMDA açıkça geçmiyorsa varsayma. "
        "Bağlam dışıysa 'Bilmiyorum' de."
    )
    return (
        f"{system}{history_context}\n\n" 
        f"Kullanıcı sorusu: {question}\n\n" 
        f"Bağlam (Soru-Cevap parçaları):\n{context_block}\n\n" 
        f"Cevap (en fazla 4-5 cümle):"
    )


def answer_question(question: str, top_k: int = 5, chat_history: List[dict] = None) -> dict:
    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "meta.jsonl"
    retriever = FaissRetriever(index_path=index_path, meta_path=meta_path, embedding_model_name=EMBEDDING_MODEL_NAME)
    contexts = retriever.search(question, top_k=top_k)
    prompt = format_prompt(question, contexts, chat_history)
    answer = generate_answer(prompt, question, contexts)

    # Basit kaynak ipucu çıkarımı (örn. "madde 3", "124. madde")
    citation = None
    try:
        merged = " \n ".join([c for c, _ in contexts]).lower()
        m = re.search(r"(\d{1,3})\.?\s*madde", merged)
        if not m:
            m = re.search(r"madde\s*(\d{1,3})", merged)
        if m:
            citation = f"Anayasa madde {m.group(1)}"
    except Exception:
        citation = None

    return {"question": question, "answer": answer, "contexts": contexts, "source_hint": citation}



