import os
import shutil
from datetime import datetime
from typing import List, Tuple, Optional

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import secrets
import easyocr

# Optional import: PyMuPDF for PDF → images
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# =========================
# Config (env-overridable)
# =========================
APP_HOST = os.environ.get('EASYOCR_HOST', '0.0.0.0')  # bind to all interfaces for LAN
APP_PORT = int(os.environ.get('EASYOCR_PORT', '7860'))
UPLOAD_DIR = os.environ.get('EASYOCR_UPLOAD_DIR', 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp', 'pdf'}
MAX_CONTENT_LENGTH = int(os.environ.get('EASYOCR_MAX_MB', '32')) * 1024 * 1024  # MB → bytes
DEFAULT_LANGS = os.environ.get('EASYOCR_LANGS', 'en,sk,cs,de').split(',')

WAITRESS_THREADS = int(os.environ.get('EASYOCR_THREADS', '8'))

# =========================
# Flask setup
# =========================
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get('EASYOCR_SECRET', secrets.token_hex(32))

_reader_cache = {}

# =========================
# Helpers
# =========================
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def try_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_reader(langs_tuple, use_gpu: bool):
    """Create or reuse EasyOCR reader (cached per language + GPU mode)."""
    key = (tuple(sorted(langs_tuple)), bool(use_gpu))
    if key in _reader_cache:
        return _reader_cache[key]
    try:
        reader = easyocr.Reader(list(langs_tuple), gpu=use_gpu)
    except Exception:
        # Fallback to CPU if GPU init fails
        if use_gpu:
            reader = easyocr.Reader(list(langs_tuple), gpu=False)
        else:
            raise
    _reader_cache[key] = reader
    return reader


def parse_page_range(range_str: str, total_pages: int) -> List[int]:
    """
    Parse page range like "1-3,5" into 0-based page indices.
    If empty/invalid, return all pages.
    """
    if not range_str:
        return list(range(total_pages))
    pages: List[int] = []
    parts = [p.strip() for p in range_str.split(',') if p.strip()]
    for part in parts:
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                start = max(1, int(a))
                end = min(total_pages, int(b))
                if start <= end:
                    pages.extend(list(range(start - 1, end)))
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.append(p - 1)
            except ValueError:
                continue
    pages = sorted(set([p for p in pages if 0 <= p < total_pages]))
    return pages or list(range(total_pages))


def render_pdf_to_images(pdf_path: str, dpi: int, out_base: str) -> Tuple[List[str], Optional[str]]:
    """
    Render selected pages of a PDF into PNG images. Returns list of image paths and an optional warning string.
    Requires PyMuPDF (pymupdf).
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is not installed. Install with: pip install pymupdf")

    warn = None
    doc = fitz.open(pdf_path)
    total = doc.page_count

    # Read form inputs for range via request (Flask context)
    range_str = request.form.get('page_range', '').strip()
    sel_pages = parse_page_range(range_str, total)

    # DPI to scale
    try:
        dpi_val = int(request.form.get('dpi', '200'))
        dpi_val = max(72, min(600, dpi_val))
    except Exception:
        dpi_val = 200
    scale = dpi_val / 72.0
    mat = fitz.Matrix(scale, scale)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    img_paths: List[str] = []
    for i in sel_pages:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        fn = f"{out_base}-p{str(i+1).zfill(4)}.png"
        out_path = os.path.join(UPLOAD_DIR, fn)
        pix.save(out_path)
        img_paths.append(out_path)
    doc.close()

    if len(img_paths) > 50:
        warn = f"Rendered {len(img_paths)} pages; this may be slow."

    return img_paths, warn


def ocr_images(reader, image_paths: List[str]):
    lines = []
    for img_path in image_paths:
        results = reader.readtext(img_path, detail=1, paragraph=False)
        for (bbox, text, conf) in results:
            if text:
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                lines.append({'text': text, 'conf': conf_f})
    return lines


def purge_environment(keep_uploads: bool = True):
    """
    Reset in-memory caches. If keep_uploads=True (default), do NOT delete uploads.
    """
    # Clear EasyOCR reader cache
    _reader_cache.clear()

    if not keep_uploads:
        # Not used by /reset anymore, but kept for potential future hard reset.
        try:
            if os.path.isdir(UPLOAD_DIR):
                for name in os.listdir(UPLOAD_DIR):
                    p = os.path.join(UPLOAD_DIR, name)
                    try:
                        if os.path.isfile(p) or os.path.islink(p):
                            os.unlink(p)
                        elif os.path.isdir(p):
                            shutil.rmtree(p)
                    except Exception:
                        pass
            else:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
        except Exception:
            pass


# =========================
# Routes
# =========================
@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        result=None,
        image_url=None,
        cuda_ok=try_cuda(),
        plain_text='',
        text_url=None,
        is_pdf=False,
        page_range='',
        dpi_default=200,
        warn=None
    )


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Unsupported file type')
        return redirect(url_for('index'))

    langs = request.form.getlist('langs') or DEFAULT_LANGS
    langs = [l.strip() for l in langs if l.strip()]
    use_gpu = request.form.get('gpu') == 'on'

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
    ext = file.filename.rsplit('.', 1)[1].lower()
    safe_name = f'input-{ts}.{ext}'
    path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(path)

    is_pdf = ext == 'pdf'
    warn = None

    image_paths: List[str] = []
    preview_url = None

    if is_pdf:
        try:
            base_no_ext = os.path.splitext(safe_name)[0]
            image_paths, warn = render_pdf_to_images(path, dpi=200, out_base=base_no_ext)
            if not image_paths:
                flash('No pages rendered from PDF.')
                return redirect(url_for('index'))
            preview_first = os.path.basename(image_paths[0])
            preview_url = url_for('uploaded_file', filename=preview_first)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('index'))
    else:
        image_paths = [path]
        preview_url = url_for('uploaded_file', filename=safe_name)

    # OCR
    reader = get_reader(tuple(langs), use_gpu)
    lines = ocr_images(reader, image_paths)

    # Plain text + save as .txt
    plain_text = '\n'.join([l['text'] for l in lines])
    txt_filename = f"{os.path.splitext(safe_name)[0]}.txt"
    txt_path = os.path.join(UPLOAD_DIR, txt_filename)
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(plain_text)
        text_url = url_for('uploaded_file', filename=os.path.basename(txt_path))
    except Exception:
        text_url = None

    return render_template(
        'index.html',
        result={'lines': lines, 'count': len(lines), 'langs': langs, 'gpu': use_gpu},
        image_url=preview_url,
        cuda_ok=try_cuda(),
        plain_text=plain_text,
        text_url=text_url,
        is_pdf=is_pdf,
        page_range=request.form.get('page_range', ''),
        dpi_default=request.form.get('dpi', '200'),
        warn=warn
    )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.route('/reset', methods=['POST'])
def reset():
    # Only clear caches/state; DO NOT delete uploads.
    purge_environment(keep_uploads=True)
    flash('Reset: cache cleared and page state reset. Uploads were kept.')
    return redirect(url_for('index'))


# =========================
# Main entry
# =========================
if __name__ == '__main__':
    from waitress import serve
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"Starting EasyOCR GUI on http://{APP_HOST}:{APP_PORT}")
    serve(app, host=APP_HOST, port=APP_PORT, threads=WAITRESS_THREADS)
