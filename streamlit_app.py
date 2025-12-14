import os
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("seu-chatbot")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

client = OpenAI(api_key=OPENAI_API_KEY)

def pinecone_search(query, k=5):
    q_emb = model.encode(query).tolist()

    res = index.query(
        vector=q_emb,
        top_k=k,
        include_metadata=True
    )

    results = []
    for m in res["matches"]:
        results.append({
            "score": m["score"],
            "text": m["metadata"]["text"],
            "source": m["metadata"]["source"]
        })

    return results

# RAG + LLM
def build_context(results):
    ctx = ""
    for r in results:
        ctx += f"\n\n--- من المصدر: {r['source']} ---\n{r['text']}"
    return ctx


def answer_with_llm(query, results):
    prompt = f"""
أجب على السؤال التالي باستخدام النصوص فقط.
إذا لم تجد الإجابة، قل:
"المعلومة غير موجودة في المستندات".

السؤال:
{query}

النصوص:
{build_context(results)}

الإجابة:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت مساعد جامعي دقيق جدًا."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

st.set_page_config(page_title="SEU Chatbot", page_icon="🎓")

st.title("SEU Chatbot")

query = st.text_input("اكتب سؤالك هنا")

if st.button("إرسال"):
    if not query.strip():
        st.warning("اكتب سؤال أولًا")
    else:
        with st.spinner("جاري البحث في المستندات..."):
            results = pinecone_search(query, k=5)
            answer = answer_with_llm(query, results)

        st.subheader("الإجابة")
        st.write(answer)

        # st.markdown("---")
        # with st.expander("📄 النصوص المستخدمة"):
        #     for r in results:
        #         st.write(f"**{r['source']}** — Score: {r['score']:.3f}")
        #         st.write(r["text"][:700])
        #         st.write("---")



