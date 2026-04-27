import fitz
import pdfplumber
import os
import re
import time
import numpy as np
from PIL import Image
import layoutparser as lp

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_IMG_DIR = "images"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

USE_LAYOUT = True

# -----------------------------
# LOAD LAYOUT MODEL
# -----------------------------
def load_layout_model():
    try:
        model = lp.AutoLayoutModel(
            "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config"
        )
        print("[DEBUG] Layout model loaded ✅")
        return model
    except Exception as e:
        print("[DEBUG] Layout model failed ❌:", e)
        return None

layout_model = load_layout_model()

def log(msg):
    print(f"[DEBUG] {msg}")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    # Fix hyphen line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -----------------------------
# NORMAL EXTRACTION
# -----------------------------
def extract_blocks_normal(page):
    blocks = page.get_text("blocks")
    cleaned = []

    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        if text.strip():
            cleaned.append({
                "content": text,
                "bbox": (x0, y0, x1, y1)
            })

    return cleaned

# -----------------------------
# COMPLEX DETECTION
# -----------------------------
def is_complex_layout(blocks):
    if not blocks:
        return False

    if len(blocks) > 25:
        return True

    xs = [b["bbox"][0] for b in blocks]
    if not xs:
        return False

    if max(xs) - min(xs) > 250:
        return True

    return False

# -----------------------------
# LAYOUT PARSER
# -----------------------------
def process_layout(page):
    if layout_model is None:
        return extract_blocks_normal(page)

    try:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        layout = layout_model.detect(img)

        blocks = []
        for b in layout:
            if b.type in ["Text", "Title"]:
                x1, y1, x2, y2 = map(int, b.coordinates)
                text = page.get_textbox((x1, y1, x2, y2))

                if text.strip():
                    blocks.append({
                        "content": text,
                        "bbox": (x1, y1, x2, y2)
                    })

        return blocks if blocks else extract_blocks_normal(page)

    except Exception as e:
        log(f"Layout failed → {e}")
        return extract_blocks_normal(page)

# -----------------------------
# COLUMN DETECTION
# -----------------------------
def detect_columns(blocks):
    if not blocks:
        return [blocks]

    centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]
    mid = np.median(centers)

    left, right = [], []

    for b in blocks:
        center = (b["bbox"][0] + b["bbox"][2]) / 2
        if center < mid:
            left.append(b)
        else:
            right.append(b)

    left = sorted(left, key=lambda b: b["bbox"][1])
    right = sorted(right, key=lambda b: b["bbox"][1])

    # If one column empty → fallback
    if not left or not right:
        return [sorted(blocks, key=lambda b: b["bbox"][1])]

    return [left, right]

# -----------------------------
# 🔥 COLUMN MERGE (KEY FIX)
# -----------------------------
def merge_columns(columns):
    if len(columns) == 1:
        return columns[0]

    left, right = columns
    merged = []

    # Step 1: full left column
    for b in left:
        merged.append(b)

    # Step 2: merge right column smartly
    for b in right:
        if merged:
            prev = clean_text(merged[-1]["content"])
            curr = clean_text(b["content"])

            # If sentence incomplete → merge
            if prev and not prev.endswith(('.', '!', '?', ':')):
                merged[-1]["content"] += " " + curr
            else:
                merged.append(b)
        else:
            merged.append(b)

    return merged

# -----------------------------
# PAGE PROCESSING
# -----------------------------
def process_page(page):
    blocks = extract_blocks_normal(page)

    if not blocks:
        return []

    if USE_LAYOUT and is_complex_layout(blocks):
        log("→ Using LayoutParser")
        blocks = process_layout(page)

    # 🔥 CRITICAL FIX
    columns = detect_columns(blocks)
    blocks = merge_columns(columns)

    return blocks

# -----------------------------
# TABLES
# -----------------------------
def extract_tables(pdf_path):
    tables_data = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                tables_data[i+1] = tables

    return tables_data

def table_to_md(table):
    header = [str(x) for x in table[0]]

    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"

    for row in table[1:]:
        md += "| " + " | ".join([str(x) for x in row]) + " |\n"

    return md

# -----------------------------
# TYPE DETECTION
# -----------------------------
def detect_type(text):
    t = text.lower()

    if "warning" in t:
        return "warning"

    if len(text) < 80 and text.isupper():
        return "heading"

    return "paragraph"

# -----------------------------
# GROUPING
# -----------------------------
def group_blocks(blocks):
    grouped = []
    buffer = ""

    for b in blocks:
        text = clean_text(b["content"])
        if not text:
            continue

        t = detect_type(text)

        if t in ["heading", "warning"]:
            if buffer:
                grouped.append(("paragraph", buffer.strip()))
                buffer = ""
            grouped.append((t, text))
        else:
            buffer += " " + text

    if buffer:
        grouped.append(("paragraph", buffer.strip()))

    return grouped

# -----------------------------
# MARKDOWN BUILDER
# -----------------------------
def build_md(pages, tables):
    md = ["# Extracted Document\n"]

    for p in pages:
        md.append(f"\n---\n## Page {p['page']}\n")

        grouped = group_blocks(p["blocks"])

        for t, text in grouped:
            if t == "heading":
                md.append(f"\n### {text}\n")
            elif t == "warning":
                md.append(f"\n> ⚠️ {text}\n")
            else:
                md.append(text + "\n")

        if p["page"] in tables:
            for table in tables[p["page"]]:
                md.append("\n#### Table\n")
                md.append(table_to_md(table))

        if p["images"]:
            md.append("\n#### Images\n")
            for img in p["images"]:
                md.append(f"![image]({img.replace(chr(92),'/')})\n")

    return "\n".join(md)

# -----------------------------
# MAIN
# -----------------------------
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        log(f"Processing page {i+1}")

        blocks = process_page(page)

        # Images
        images = []
        for j, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)

            name = f"page{i+1}_img{j+1}.png"
            path = os.path.join(OUTPUT_IMG_DIR, name)

            with open(path, "wb") as f:
                f.write(base["image"])

            images.append(path)

        pages.append({
            "page": i+1,
            "blocks": blocks,
            "images": images
        })

    return pages

# -----------------------------
# RUN
# -----------------------------
def parse_pdf(pdf_path):
    start = time.time()

    pages = extract_pages(pdf_path)
    tables = extract_tables(pdf_path)

    md = build_md(pages, tables)

    with open("output_v12.md", "w", encoding="utf-8") as f:
        f.write(md)

    log(f"✅ Done in {round(time.time()-start,2)} sec")

if __name__ == "__main__":
    parse_pdf("pdf_file.pdf")