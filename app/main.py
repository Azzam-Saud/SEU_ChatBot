from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag_engine import rag_llm_answer

app = FastAPI(title="SEU RAG API")

# ربط ملفات الواجهة
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
def ask(q: Query):
    return {"answer": rag_llm_answer(q.question)}