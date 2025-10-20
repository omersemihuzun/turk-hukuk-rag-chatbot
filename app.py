import streamlit as st
import re

from src.rag_pipeline import answer_question


st.set_page_config(page_title="BüyükSinema RAG Chatbot", page_icon="🎬", layout="wide")
st.title("⚖️ Türk Hukuk Chatbot (RAG)")
st.caption("Renicames/turkish-law-chatbot veri setiyle RAG tabanlı danışman")

with st.sidebar:
    st.header("Bilgi")
    st.markdown("Bu araç, hukuk Soru-Cevap parçalarından bağlam çekip kısa yanıt üretir.")

# Basit sohbet durumu
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Merhaba! Hukuki sorunuzu yazın, bağlamdan kısa ve net yanıt vereyim."}
    ]

# Önceki mesajları göster
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _strip_local_summary(answer: str) -> str:
    if not answer:
        return answer
    # '(Yerel özet) ...' bloğunu kaldır
    if answer.lstrip().startswith("(Yerel özet)"):
        # ilk satırı atla
        parts = answer.split("\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return ""
    # "Soru:" / "Cevap:" ile başlayan satırları gizle
    lines = [ln for ln in answer.split("\n") if not re.match(r"^\s*(Soru|Cevap)\s*:\s*", ln)]
    cleaned = "\n".join(lines).strip()
    return cleaned

user_input = st.chat_input(placeholder="Örn: Anayasa madde 1 neyi düzenler?")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Cevap üzerinde çalışılıyor..."):
            resp = answer_question(user_input, top_k=5, chat_history=st.session_state["messages"])
        final_answer = _strip_local_summary(resp.get("answer", ""))
        source = resp.get("source_hint")
        if source:
            final_answer = f"{final_answer}\n\nKaynak: {source}"
        st.markdown(final_answer)
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})



