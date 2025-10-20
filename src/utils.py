"""
Yardımcı fonksiyonlar:
- chunk_text: Uzun metinleri sabit boyutlu parçalara böler (indeksleme için).
- batched: Iterable'ı belirli büyüklükte gruplar halinde verir (verimli encoding için).
"""
from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, max_chars: int = 512) -> List[str]:
    """Basit karakter tabanlı bölme. Kelime/sentence aware gerekmeyen hızlı çözüm."""
    if not text:
        return []
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        chunks.append(cleaned[start:end])
        start = end
    return chunks


def batched(iterable: Iterable, batch_size: int) -> Iterable[list]:
    """Verilen iterable'ı batch_size büyüklüğünde listeler halinde döndürür."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch



