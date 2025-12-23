import os, pickle, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from loaders import extract_txt_from_files, extract_word, extract_excel
from chunking import chunk_policy_qna_articles

DIR1 = r"D:\Azzam\Personal_Projects\SEU\filtered_data\Word_Excel"
DIR2 = r"D:\Azzam\Personal_Projects\SEU\filtered_data\txt"

records = []

def process_dir(folder):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        ext = fname.split(".")[-1].lower()

        if ext == "txt":
            recs = extract_txt_from_files(path, fname)
        elif ext == "docx":
            recs = extract_word(path, fname)
        elif ext in ["xlsx", "xls"]:
            recs = extract_excel(path, fname)
        else:
            continue

        records.extend(recs)

process_dir(DIR1)
process_dir(DIR2)

chunked = []

for rec in records:
    chunks = [rec["text"]] if rec["type"] == "excel" else chunk_policy_qna_articles(rec["text"])
    for i, ch in enumerate(chunks):
        chunked.append({
            "id": f'{rec["id"]}__chunk_{i}',
            "text": ch
        })

texts = [c["text"] for c in chunked]
ids = [c["id"] for c in chunked]

model = SentenceTransformer("intfloat/multilingual-e5-large")
embs = model.encode(["passage: " + t for t in texts], show_progress_bar=True).astype("float32")

faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, "faiss.index")
pickle.dump({"texts": texts, "ids": ids}, open("meta.pkl", "wb"))

print("✅ Index built successfully")