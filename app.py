import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from functools import lru_cache
from multiprocessing import Pool, cpu_count

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash
)
from werkzeug.utils import secure_filename

import secrets
import easyocr

# Optional PDF support
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# =========================
# Configuration
# =========================

APP_HOST = os.environ.get("EASYOCR_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("EASYOCR_PORT", "7860"))
UPLOAD_DIR = os.environ.get("EASYOCR_UPLOAD_DIR", "uploads")

MAX_PDF_MB = 50
MAX_IMG_MB = 10
MAX_PDF_PAGES = 50

CONFIDENCE_THRESHOLD = 0.4

DEFAULT_LANGS = ["sk"]
UI_LANG_DEFAULT = "sk"
SUPPORTED_UI_LANGS = {"sk", "en"}

ALLOWED_EXT = {"png", "jpg", "jpeg", "pdf"}
WAITRESS_THREADS = int(os.environ.get("EASYOCR_THREADS", "8"))

# =========================
# Flask setup
# =========================

app = Flask(__name__)
app.secret_key = os.environ.get("EASYOCR_SECRET", secrets.token_hex(32))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# Helpers
# =========================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def validate_file(file) -> Optional[str]:
    """
    Validate uploaded file size and type.
    """
    ext = file.filename.rsplit(".", 1)[1].lower()
    size = request.content_length or 0

    if ext == "pdf" and size > MAX_PDF_MB * 1024 * 1024:
        return f"PDF exceeds {MAX_PDF_MB} MB limit"
    if ext in {"png", "jpg", "jpeg"} and size > MAX_IMG_MB * 1024 * 1024:
        return f"Image exceeds {MAX_IMG_MB} MB limit"
    return None


def try_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


CUDA_OK = try_cuda()


@lru_cache(maxsize=4)
def get_reader_cached(langs: Tuple[str], use_gpu: bool):
    """
    Cached EasyOCR reader with bounded LRU cache.
    """
    try:
        return easyocr.Reader(list(langs), gpu=use_gpu)
    except Exception:
        if use_gpu:
            return easyocr.Reader(list(langs), gpu=False)
        raise


def parse_page_range(range_str: str, total_pages: int) -> List[int]:
    """
    Parse page range string into 0-based indices.
    """
    if not range_str:
        return list(range(total_pages))

    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = map(int, part.split("-", 1))
                for p in range(max(1, a), min(total_pages, b) + 1):
                    pages.add(p - 1)
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
            except ValueError:
                continue

    return sorted(pages)


def render_pdf_to_images(
    pdf_path: str,
    dpi: int,
    out_base: str,
    page_range: str
) -> Tuple[List[Tuple[int, str]], Optional[str]]:
    """
    Render PDF pages into PNG images.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed")

    doc = fitz.open(pdf_path)
    total_pages = min(doc.page_count, MAX_PDF_PAGES)

    selected_pages = parse_page_range(page_range, total_pages)
    selected_pages = selected_pages[:MAX_PDF_PAGES]

    scale = max(72, min(dpi, 600)) / 72.0
    mat = fitz.Matrix(scale, scale)

    results = []

    for i in selected_pages:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        name = f"{out_base}-p{str(i+1).zfill(4)}.png"
        path = os.path.join(UPLOAD_DIR, name)
        pix.save(path)
        results.append((i + 1, path))

    doc.close()

    warn = None
    if len(results) >= MAX_PDF_PAGES:
        warn = f"PDF page limit reached ({MAX_PDF_PAGES})"

    return results, warn


def ocr_single(args):
    """
    OCR a single image (CPU multiprocessing worker).
    """
    reader, page_no, img_path = args
    results = reader.readtext(img_path, detail=1)
    lines = []

    for _, text, conf in results:
        if text and float(conf) >= CONFIDENCE_THRESHOLD:
            lines.append(text)

    return page_no, lines


def ocr_images(
    reader,
    images: List[Tuple[int, str]],
    use_gpu: bool
):
    """
    Perform OCR using GPU (single process) or CPU (multiprocessing).
    """
    if use_gpu:
        output = []
        for page_no, path in images:
            results = reader.readtext(path, detail=1)
            lines = [
                text for _, text, conf in results
                if text and float(conf) >= CONFIDENCE_THRESHOLD
            ]
            output.append((page_no, lines))
        return output

    with Pool(min(cpu_count(), 4)) as pool:
        return pool.map(
            ocr_single,
            [(reader, p, path) for p, path in images]
        )


# =========================
# Routes
# =========================

@app.route("/", methods=["GET"])
def index():
    ui_lang = request.args.get("lang", UI_LANG_DEFAULT)
    if ui_lang not in SUPPORTED_UI_LANGS:
        ui_lang = UI_LANG_DEFAULT

    return render_template(
        "index.html",
        ui_lang=ui_lang,
        cuda_ok=CUDA_OK,
        result=None
    )


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        flash("No file provided")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type")
        return redirect(url_for("index"))

    err = validate_file(file)
    if err:
        flash(err)
        return redirect(url_for("index"))

    langs = request.form.getlist("langs") or DEFAULT_LANGS
    use_gpu = request.form.get("gpu") == "on"

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = secure_filename(f"input-{ts}.{ext}")
    path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(path)

    flash("Processing started...")

    images = []
    warn = None

    if ext == "pdf":
        images, warn = render_pdf_to_images(
            path,
            dpi=int(request.form.get("dpi", 200)),
            out_base=os.path.splitext(safe_name)[0],
            page_range=request.form.get("page_range", "")
        )
    else:
        images = [(1, path)]

    reader = get_reader_cached(tuple(sorted(langs)), use_gpu)
    ocr_result = ocr_images(reader, images, use_gpu)

    text_output = []
    for page_no, lines in sorted(ocr_result):
        text_output.append(f"--- Page {page_no} ---")
        text_output.extend(lines)
        text_output.append("")

    plain_text = "\n".join(text_output)

    txt_name = secure_filename(os.path.splitext(safe_name)[0] + ".txt")
    txt_path = os.path.join(UPLOAD_DIR, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(plain_text)

    return render_template(
        "index.html",
        result={
            "text": plain_text,
            "pages": len(images),
            "langs": langs,
            "gpu": use_gpu
        },
        text_url=url_for("uploaded_file", filename=txt_name),
        warn=warn,
        cuda_ok=CUDA_OK
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_DIR,
        secure_filename(filename),
        as_attachment=False
    )


@app.route("/reset", methods=["POST"])
def reset():
    """
    Reset runtime caches without deleting uploads.
    """
    get_reader_cached.cache_clear()
    flash("Application cache cleared.")
    return redirect(url_for("index"))


# =========================
# Main
# =========================

if __name__ == "__main__":
    from waitress import serve
    logger.info("Starting EasyOCR GUI")
    serve(app, host=APP_HOST, port=APP_PORT, threads=WAITRESS_THREADS)
