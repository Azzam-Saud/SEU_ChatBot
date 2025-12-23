import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.config import OPENAI_API_KEY

model = SentenceTransformer("intfloat/multilingual-e5-large")
index = faiss.read_index("faiss.index")

meta = pickle.load(open("meta.pkl", "rb"))
texts, ids = meta["texts"], meta["ids"]

client = OpenAI(api_key=OPENAI_API_KEY)

def search(query, k=15):
    q_emb = model.encode(["query: " + query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    return [{"score": float(D[0][i]), "text": texts[idx]} for i, idx in enumerate(I[0])]

def rag_llm_answer(query: str):
    results = search(query)
    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
أجب فقط من النص التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()
