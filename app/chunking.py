import re

def is_question_line(line: str) -> bool:
    if not line:
        return False
    if line.endswith("؟") or line.endswith("?"):
        return True
    starters = ("ما", "ماهي", "ما هي", "كيف", "هل", "كم", "متى", "لماذا", "أين", "وش")
    return any(line.startswith(s) for s in starters)

def is_article_header(line: str) -> bool:
    return bool(re.match(r"^(المادة)\s*[\(\[]?\s*\d+|^(المادة)\s+\S+", line))

def word_chunk(text: str, max_words=250, overlap_words=80):
    words = text.split()
    chunks, start = [], 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap_words

    return chunks

def chunk_policy_qna_articles(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, buffer = [], []
    mode = "normal"

    def flush_normal(buf):
        if not buf:
            return
        block = "\n".join(buf)
        if len(block.split()) > 350:
            chunks.extend(word_chunk(block))
        else:
            chunks.append(block)

    def flush(buf):
        if buf:
            chunks.append("\n".join(buf))

    for line in lines:
        if is_article_header(line):
            flush_normal(buffer) if mode == "normal" else flush(buffer)
            buffer, mode = [line], "article"
            continue

        if is_question_line(line):
            flush_normal(buffer) if mode == "normal" else flush(buffer)
            buffer, mode = [line], "qna"
            continue

        buffer.append(line)

    flush_normal(buffer) if mode == "normal" else flush(buffer)
    return chunks