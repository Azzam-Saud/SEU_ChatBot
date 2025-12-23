import docx
import pandas as pd
from docx.table import Table
from docx.text.paragraph import Paragraph

# ---------- TXT ----------
def extract_txt_from_files(path: str, filename: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []

    if not text.strip():
        return []

    return [{
        "id": f"{filename}__full",
        "source": filename,
        "type": "txt",
        "text": text.strip()
    }]

# ---------- Word ----------
def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('}tbl'):
            yield Table(child, parent)

def extract_word(path: str, filename: str):
    try:
        doc = docx.Document(path)
    except Exception:
        return []

    blocks = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                blocks.append(text)

        elif isinstance(block, Table):
            blocks.append("")
            for row in block.rows:
                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                )
                if row_text:
                    blocks.append(row_text)
            blocks.append("")

    if not blocks:
        return []

    return [{
        "id": f"{filename}__doc",
        "source": filename,
        "type": "word",
        "text": "\n".join(blocks)
    }]

# ---------- Excel ----------
def extract_excel(path, filename):
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except:
        return []

    output = []

    for sheet, df in sheets.items():
        if df.empty:
            continue

        headers = list(df.columns)

        for idx, row in df.iterrows():
            fields = []
            for h, v in zip(headers, row.tolist()):
                v = str(v).strip()
                if v.lower() == "nan" or v == "":
                    continue
                fields.append(f"{h}: {v}")

            if fields:
                output.append({
                    "id": f"{filename}__sheet_{sheet}__row_{idx}",
                    "source": filename,
                    "type": "excel",
                    "text": "\n".join(fields)
                })

    return output