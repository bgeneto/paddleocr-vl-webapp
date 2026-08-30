"""
PaddleOCR-VL Document Parser Streamlit Application

This application provides a web interface for document OCR using PaddleOCR-VL
with vLLM backend for production-ready inference.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF for PDF preview
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# Configuration from environment variables
APP_TITLE = os.getenv("APP_TITLE", "PaddleOCR-VL Document Parser")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "Upload PDF or image files to convert them to Markdown using PaddleOCR-VL with vLLM backend",
)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "99"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "250"))
# Parallel workers - keep low since API may serialize requests anyway
MAX_PARALLEL_PAGES = int(os.getenv("MAX_PARALLEL_PAGES", "8"))
MAX_PREVIEW_PAGES = int(os.getenv("MAX_PREVIEW_PAGES", "10"))  # Limit preview rendering
# Pages per chunk - HIGHER = better GPU batching (vLLM processes all pages in chunk together)
# This is the KEY setting for GPU utilization. Increase if you have enough VRAM.
PAGES_PER_CHUNK = int(os.getenv("PAGES_PER_CHUNK", "16"))  # Pages per API request for GPU batching

# PaddleOCR-VL API Configuration
PADDLEOCR_VL_API_URL = os.getenv(
    "PADDLEOCR_VL_API_URL", "http://paddleocr-vl-api:8080/layout-parsing"
)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))

# Processing options
USE_DOC_ORIENTATION_CLASSIFY = (
    os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false").lower() == "true"
)
USE_DOC_UNWARPING = os.getenv("USE_DOC_UNWARPING", "false").lower() == "true"
USE_LAYOUT_DETECTION = os.getenv("USE_LAYOUT_DETECTION", "true").lower() == "true"
USE_CHART_RECOGNITION = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"
PRETTIFY_MARKDOWN = os.getenv("PRETTIFY_MARKDOWN", "true").lower() == "true"
VISUALIZE_RESULTS = os.getenv("VISUALIZE_RESULTS", "false").lower() == "true"
# Speed-first remains the default. true selects the quality YAML (compose),
# high-recall infer fields, and post-chunk /restructure-pages.
OCR_QUALITY_FIRST = os.getenv("OCR_QUALITY_FIRST", "false").lower() == "true"
_QUALITY_VLM_MAX_PIXELS = 1_605_632

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]

# =============================================================================
# Persistence / result cache
# =============================================================================
# Successful results are stored under DATA_DIR, keyed by the SHA-256 of the
# file content and a fingerprint of the six processing options:
#   DATA_DIR/<sha256>/<options-fingerprint>/{meta.json,result.md,result.zip,pages/}
# Re-uploading a byte-identical file with the same options is served instantly
# from disk (no API call). In-progress PDF jobs also checkpoint each completed
# page under pages/{n:04d}.json so a later Start OCR can skip already-OCR'd
# pages. The directory survives container recreation via the ./data bind mount
# (see compose.yaml).
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# (option key, short name) pairs in fixed order; defines the fingerprint format
_OPTION_FINGERPRINT_KEYS = (
    ("use_doc_orientation_classify", "orient"),
    ("use_doc_unwarping", "unwarp"),
    ("use_layout_detection", "layout"),
    ("use_chart_recognition", "chart"),
    ("prettify_markdown", "pretty"),
    ("visualize", "vis"),
)


def compute_sha256(content: bytes) -> str:
    """Compute the SHA-256 hex digest of file content (already fully in memory)."""
    return hashlib.sha256(content).hexdigest()


def options_fingerprint(options: dict) -> str:
    """Deterministic compact fingerprint of the six processing options.

    Example: 'orient=1_unwarp=0_layout=1_chart=0_pretty=1_vis=0'. Key order is
    fixed here, so the ordering of the caller's dict does not matter.
    Quality-first runs append '_q=1' so they never reuse a speed-first cache
    entry; speed-first fingerprints are unchanged.
    """
    fingerprint = "_".join(
        f"{short}={'1' if options.get(key) else '0'}"
        for key, short in _OPTION_FINGERPRINT_KEYS
    )
    if OCR_QUALITY_FIRST:
        return f"{fingerprint}_q=1"
    return fingerprint


def cache_dirs(hash_str: str, fingerprint: str) -> Path:
    """Return (creating if needed) the cache entry directory for a pair."""
    entry_dir = DATA_DIR / hash_str / fingerprint
    entry_dir.mkdir(parents=True, exist_ok=True)
    return entry_dir


def display_stem(display_name: str) -> str:
    """Sanitized base name used for artifact filenames and ZIP structure."""
    return Path(display_name).stem.strip() or "document"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write bytes to a temp file in the same directory, then rename into place."""
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def save_result_to_disk(
    hash_str: str,
    fingerprint: str,
    display_name: str,
    markdown_text: str,
    images: dict,
    page_count: int,
    options: dict,
) -> None:
    """Persist a successful result so re-uploads can be served instantly.

    Never raises: a storage failure must not break a successful in-memory
    result (logged + surfaced as a warning instead). All files are written
    atomically (temp file + rename).
    """
    try:
        entry_dir = cache_dirs(hash_str, fingerprint)
        stem = display_stem(display_name)
        now = datetime.now(timezone.utc).isoformat()

        zip_bytes = create_download_zip(markdown_text, images, stem)
        md_bytes = markdown_text.encode("utf-8")
        created_at = now
        meta_path = entry_dir / "meta.json"
        if meta_path.is_file():
            try:
                created_at = (
                    json.loads(meta_path.read_text(encoding="utf-8")).get("created_at")
                    or now
                )
            except Exception:
                pass
        meta = {
            "display_name": display_name,
            "hash": hash_str,
            "fingerprint": fingerprint,
            "options": {key: bool(options.get(key)) for key, _ in _OPTION_FINGERPRINT_KEYS},
            "page_count": page_count,
            "status": "complete",
            "created_at": created_at,
            "updated_at": now,
            "sizes": {"md": len(md_bytes), "zip": len(zip_bytes)},
        }

        _atomic_write_bytes(entry_dir / "result.md", md_bytes)
        _atomic_write_bytes(entry_dir / "result.zip", zip_bytes)
        _atomic_write_bytes(
            entry_dir / "meta.json", json.dumps(meta, indent=2).encode("utf-8")
        )
    except Exception as e:
        logger.warning(
            "Failed to save result to disk cache (%s/%s): %s", hash_str, fingerprint, e
        )
        try:
            st.warning(f"⚠️ Could not persist result to disk cache: {e}")
        except Exception:
            pass


def load_cached_result(hash_str: str, fingerprint: str) -> dict | None:
    """Load a previously stored result, or None on any miss or corruption.

    On a hit, reconstructs the same shape used by the in-memory session cache:
    {'markdown', 'images', 'display_name', 'from_disk_cache', 'fingerprint'},
    where images keys are the relative paths ('page_N/file' or 'file') matching
    extract_markdown_from_response, with base64 values. Corrupt or partial
    entries are treated as misses and left on disk for inspection.
    """
    entry_dir = DATA_DIR / hash_str / fingerprint
    # Incomplete checkpoints (pages/*.json only) are not a full hit; resume
    # happens in process_pdf_in_batches. Avoid logging those as cache misses.
    if not all(
        (entry_dir / name).is_file()
        for name in ("meta.json", "result.md", "result.zip")
    ):
        return None

    try:
        meta = json.loads((entry_dir / "meta.json").read_text(encoding="utf-8"))
        markdown_text = normalize_markdown_math(
            (entry_dir / "result.md").read_text(encoding="utf-8")
        )
        stem = display_stem(meta["display_name"])
        images_prefix = f"{stem}_images/"

        images = {}
        with zipfile.ZipFile(entry_dir / "result.zip", "r") as zf:
            names = set(zf.namelist())
            if f"{stem}.md" not in names:
                raise ValueError(f"zip is missing the {stem}.md member")
            for name in names:
                if name.startswith(images_prefix) and not name.endswith("/"):
                    data = zf.read(name)
                    images[name[len(images_prefix):]] = base64.b64encode(data).decode("ascii")

        return {
            "markdown": markdown_text,
            "images": images,
            "display_name": meta["display_name"],
            "from_disk_cache": True,
            "fingerprint": fingerprint,
        }
    except Exception as e:
        logger.warning("Disk cache miss (%s/%s): %s", hash_str, fingerprint, e)
        return None


def pages_cache_dir(hash_str: str, fingerprint: str) -> Path:
    """Directory for per-page OCR checkpoints of a cache entry."""
    return cache_dirs(hash_str, fingerprint) / "pages"


def page_result_path(hash_str: str, fingerprint: str, page_num: int) -> Path:
    """Path to the JSON checkpoint for a 0-indexed page."""
    return pages_cache_dir(hash_str, fingerprint) / f"{page_num:04d}.json"


def save_page_result(
    hash_str: str, fingerprint: str, page_num: int, parsing_result: dict
) -> None:
    """Atomically persist one page's layoutParsingResults item. Never raises."""
    try:
        pages_dir = pages_cache_dir(hash_str, fingerprint)
        pages_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(parsing_result, ensure_ascii=False).encode("utf-8")
        _atomic_write_bytes(page_result_path(hash_str, fingerprint, page_num), payload)
    except Exception as e:
        logger.warning(
            "Failed to save page %s checkpoint (%s/%s): %s",
            page_num,
            hash_str,
            fingerprint,
            e,
        )


def load_completed_pages(hash_str: str, fingerprint: str) -> dict[int, dict]:
    """Load valid per-page checkpoints. Corrupt files are skipped (will be re-OCR'd).

    The pages/ directory is the source of truth for resume progress — not meta.json.
    """
    results: dict[int, dict] = {}
    pages_dir = DATA_DIR / hash_str / fingerprint / "pages"
    if not pages_dir.is_dir():
        return results

    for path in pages_dir.glob("*.json"):
        try:
            page_num = int(path.stem)
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping corrupt page checkpoint %s: %s", path, e)
            continue
        if isinstance(payload, dict):
            results[page_num] = payload
        else:
            logger.warning("Skipping invalid page checkpoint %s: not an object", path)
    return results


def write_partial_meta(
    hash_str: str,
    fingerprint: str,
    display_name: str,
    page_count: int,
    options: dict,
) -> None:
    """Mark a cache entry as an in-progress PDF job. Never raises.

    Not the source of truth for which pages are done (see pages/*.json).
    """
    try:
        entry_dir = cache_dirs(hash_str, fingerprint)
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        meta_path = entry_dir / "meta.json"
        if meta_path.is_file():
            try:
                created_at = (
                    json.loads(meta_path.read_text(encoding="utf-8")).get("created_at")
                    or now
                )
            except Exception:
                pass
        meta = {
            "display_name": display_name,
            "hash": hash_str,
            "fingerprint": fingerprint,
            "options": {key: bool(options.get(key)) for key, _ in _OPTION_FINGERPRINT_KEYS},
            "page_count": page_count,
            "status": "partial",
            "created_at": created_at,
            "updated_at": now,
        }
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))
    except Exception as e:
        logger.warning(
            "Failed to write partial meta (%s/%s): %s", hash_str, fingerprint, e
        )


class CancellationError(Exception):
    """Raised when processing is cancelled by the user."""

    pass


# HTTP Session with connection pooling for better performance
def create_http_session() -> requests.Session:
    """Create an HTTP session with connection pooling and retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )

    # Configure connection pooling - increase pool size for parallel requests
    adapter = HTTPAdapter(
        pool_connections=MAX_PARALLEL_PAGES + 2,
        pool_maxsize=MAX_PARALLEL_PAGES + 2,
        max_retries=retry_strategy,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Global session for connection reuse
_http_session: requests.Session | None = None


def get_http_session() -> requests.Session:
    """Get or create the global HTTP session."""
    global _http_session
    if _http_session is None:
        _http_session = create_http_session()
    return _http_session


def check_api_health() -> bool:
    """Check if the PaddleOCR-VL API service is healthy."""
    try:
        # Extract base URL from the API URL
        base_url = PADDLEOCR_VL_API_URL.rsplit("/", 1)[0]
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def encode_file_to_base64(file_content: bytes) -> str:
    """Encode file content to base64 string."""
    return base64.b64encode(file_content).decode("ascii")


def apply_quality_infer_fields(payload: dict) -> dict:
    """Add modal-paddleocr infer fields when OCR_QUALITY_FIRST is on.

    Sidebar flags (orientation, unwarping, layout, chart) stay in *payload*
    as the caller set them. This only adds fields the speed UI does not expose.
    """
    if not OCR_QUALITY_FIRST:
        return payload
    payload.update(
        {
            "useSealRecognition": True,
            "useOcrForImageBlock": True,
            "markdownIgnoreLabels": [],
            "layoutThreshold": 0.2,
            "layoutNms": False,
            "layoutUnclipRatio": [1.08, 1.12],
            "layoutMergeBboxesMode": "union",
            "layoutShapeMode": "poly",
            "maxPixels": _QUALITY_VLM_MAX_PIXELS,
            "vlmExtraArgs": {
                "ocr_max_pixels": _QUALITY_VLM_MAX_PIXELS,
                "table_max_pixels": _QUALITY_VLM_MAX_PIXELS,
                "formula_max_pixels": _QUALITY_VLM_MAX_PIXELS,
                "chart_max_pixels": _QUALITY_VLM_MAX_PIXELS,
                "seal_max_pixels": _QUALITY_VLM_MAX_PIXELS,
            },
        }
    )
    return payload


def restructure_parsing_results(
    parsing_results: list, prettify_markdown: bool = True
) -> list:
    """Merge tables / relevel titles / concatenate pages after chunked OCR.

    No-ops (returns the input list) unless OCR_QUALITY_FIRST is on and there
    are at least two pages with prunedResult. Failures are logged and the
    original per-page results are kept.
    """
    if not OCR_QUALITY_FIRST or len(parsing_results) < 2:
        return parsing_results

    pages = []
    for page in parsing_results:
        pruned = page.get("prunedResult")
        if not isinstance(pruned, dict):
            logger.warning(
                "Skipping /restructure-pages: a page is missing prunedResult"
            )
            return parsing_results
        item: dict = {"prunedResult": pruned}
        images = (page.get("markdown") or {}).get("images")
        if images:
            item["markdownImages"] = images
        pages.append(item)

    url = f"{PADDLEOCR_VL_API_URL.rsplit('/', 1)[0]}/restructure-pages"
    try:
        response = get_http_session().post(
            url,
            json={
                "pages": pages,
                "mergeTables": True,
                "relevelTitles": True,
                "concatenatePages": True,
                "prettifyMarkdown": prettify_markdown,
            },
            timeout=API_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            logger.warning(
                "restructure-pages failed: %s %s",
                response.status_code,
                response.text[:500],
            )
            return parsing_results
        data = response.json()
        if data.get("errorCode") not in (None, 0):
            logger.warning(
                "restructure-pages errorCode=%s %s",
                data.get("errorCode"),
                data.get("errorMsg"),
            )
            return parsing_results
        result = data.get("result") or data
        new_results = result.get("layoutParsingResults")
        if not new_results:
            logger.warning("restructure-pages returned no layoutParsingResults")
            return parsing_results
        logger.info(
            "Restructured %s pages into %s result(s)",
            len(parsing_results),
            len(new_results),
        )
        return new_results
    except Exception as e:
        logger.warning("restructure-pages error: %s", e)
        return parsing_results


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes."""
    return base64.b64decode(base64_string)


def get_file_type(filename: str) -> int:
    """Determine file type: 0 for PDF, 1 for image."""
    ext = Path(filename).suffix.lower()
    return 0 if ext == ".pdf" else 1


def validate_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file type and size."""
    if uploaded_file is None:
        return False, "No file uploaded"

    file_name = uploaded_file.name.lower()
    file_extension = Path(file_name).suffix

    if file_extension not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {file_extension}"

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return (
            False,
            f"File size ({file_size_mb:.1f}MB) exceeds maximum ({MAX_FILE_SIZE_MB}MB)",
        )

    return True, "Valid"


def get_pdf_preview(file_content: bytes, max_pages: int = 5) -> list[bytes]:
    """Generate preview images from PDF pages."""
    previews = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            # Render at 150 DPI for preview (good balance of quality and size)
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            previews.append(pix.tobytes("png"))
        doc.close()
    except Exception as e:
        st.warning(f"Could not generate PDF preview: {e}")
    return previews


def get_pdf_page_count(file_content: bytes) -> int:
    """Get the number of pages in a PDF."""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def split_pdf_into_chunks(
    file_content: bytes, pages_per_chunk: int = None
) -> list[tuple[int, int, bytes]]:
    """
    Split a PDF into chunks of multiple pages for efficient batch processing.

    This enables better GPU utilization by sending multiple pages to vLLM at once,
    allowing it to batch-process them together.

    Args:
        file_content: Raw PDF bytes
        pages_per_chunk: Number of pages per chunk (default: PAGES_PER_CHUNK)

    Returns:
        List of tuples: (start_page, end_page, chunk_pdf_bytes)
        Page numbers are 0-indexed.
    """
    if pages_per_chunk is None:
        pages_per_chunk = PAGES_PER_CHUNK

    chunks = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        total_pages = len(doc)

        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)

            # Create a new PDF with this chunk of pages
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            chunks.append((start_page, end_page, chunk_bytes))

        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to split PDF into chunks: {e}")

    return chunks


def extract_pdf_pages(file_content: bytes, page_indices: list[int]) -> bytes:
    """Build a PDF containing the given 0-indexed pages, in that order."""
    if not page_indices:
        raise ValueError("page_indices must not be empty")
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        chunk_doc = fitz.open()
        for page_num in page_indices:
            chunk_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()
        doc.close()
        return chunk_bytes
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF pages {page_indices}: {e}")


def split_pdf_into_pages(file_content: bytes) -> list[bytes]:
    """Split a PDF into individual single-page PDFs (legacy, for small PDFs)."""
    pages = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(len(doc)):
            # Create a new PDF with just this page
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            page_bytes = single_page_doc.tobytes()
            pages.append(page_bytes)
            single_page_doc.close()
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to split PDF: {e}")
    return pages


def display_file_preview(uploaded_file, file_content: bytes):
    """Display a preview of the uploaded file."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        page_count = get_pdf_page_count(file_content)
        st.caption(f"📑 PDF Document - {page_count} page(s)")

        previews = get_pdf_preview(file_content, max_pages=3)
        if previews:
            cols = st.columns(min(len(previews), 3))
            for idx, preview in enumerate(previews):
                with cols[idx]:
                    st.image(preview, caption=f"Page {idx + 1}", width="stretch")
            if page_count > 3:
                st.caption(f"... and {page_count - 3} more page(s)")
    else:
        # Image preview
        st.caption(f"🖼️ Image - {uploaded_file.size / 1024:.1f} KB")
        st.image(file_content, caption=uploaded_file.name, width="stretch")


def process_document(
    file_content: bytes,
    filename: str,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_layout_detection: bool = True,
    use_chart_recognition: bool = False,
    prettify_markdown: bool = True,
    visualize: bool = False,
) -> dict:
    """
    Process a document using the PaddleOCR-VL API.

    Args:
        file_content: Raw bytes of the file
        filename: Original filename for type detection
        use_doc_orientation_classify: Enable document orientation classification
        use_doc_unwarping: Enable document unwarping
        use_layout_detection: Enable layout detection
        use_chart_recognition: Enable chart recognition
        prettify_markdown: Whether to prettify markdown output
        visualize: Whether to return visualization images

    Returns:
        API response dictionary
    """
    # Encode file to base64
    encoded_file = encode_file_to_base64(file_content)

    # Prepare request payload
    payload = {
        "file": encoded_file,
        "fileType": get_file_type(filename),
        "useDocOrientationClassify": use_doc_orientation_classify,
        "useDocUnwarping": use_doc_unwarping,
        "useLayoutDetection": use_layout_detection,
        "useChartRecognition": use_chart_recognition,
        "prettifyMarkdown": prettify_markdown,
        "visualize": visualize,
    }
    apply_quality_infer_fields(payload)

    # Make API request using pooled session
    session = get_http_session()
    response = session.post(
        PADDLEOCR_VL_API_URL,
        json=payload,
        timeout=API_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        error_msg = response.json().get("errorMsg", "Unknown error")
        raise RuntimeError(f"API request failed: {error_msg}")

    return response.json()


def process_chunk(args: tuple) -> tuple[int, list[int], list | None, str | None]:
    """
    Process a chunk of PDF pages. Helper function for parallel chunk processing.

    Each chunk contains multiple pages, which allows vLLM to batch-process them
    together for better GPU utilization. On success, each page is checkpointed
    to disk immediately (in this worker) so a crash of the Streamlit run still
    preserves completed pages.

    Args:
        args: Tuple of (chunk_index, page_indices, chunk_content, filename,
            options, cancel_event, file_hash, fingerprint). page_indices are
            0-indexed original PDF page numbers, in the same order as pages
            in chunk_content.

    Returns:
        Tuple of (chunk_index, page_indices, parsing_results_list, error_message)
    """
    (
        chunk_index,
        page_indices,
        chunk_content,
        filename,
        options,
        cancel_event,
        file_hash,
        fingerprint,
    ) = args

    # Check for cancellation before starting
    if cancel_event and cancel_event.is_set():
        return (chunk_index, page_indices, None, "Cancelled")

    try:
        # Send multi-page chunk to API - vLLM will batch-process all pages together
        chunk_response = process_document(
            file_content=chunk_content,
            filename=filename,
            **options,
        )
        result = chunk_response.get("result", {})
        parsing_results = result.get("layoutParsingResults", [])

        if not parsing_results or len(parsing_results) != len(page_indices):
            got = 0 if not parsing_results else len(parsing_results)
            return (
                chunk_index,
                page_indices,
                None,
                f"Expected {len(page_indices)} page results, got {got}",
            )

        # Persist before returning so a dying main thread does not lose this chunk
        if file_hash and fingerprint:
            for page_num, page_result in zip(page_indices, parsing_results):
                save_page_result(file_hash, fingerprint, page_num, page_result)

        return (chunk_index, page_indices, parsing_results, None)
    except Exception as e:
        return (chunk_index, page_indices, None, str(e))


def process_pdf_in_batches(
    file_content: bytes,
    filename: str,
    progress_callback=None,
    cancel_event: threading.Event = None,
    max_workers: int = None,
    pages_per_chunk: int = None,
    file_hash: str | None = None,
    fingerprint: str | None = None,
    **options,
) -> dict:
    """
    Process a PDF document with chunked parallel processing for optimal GPU utilization.

    Instead of processing single pages, this splits the PDF into multi-page chunks
    (e.g., 8 pages each). Each chunk is sent to the API as a multi-page PDF, allowing
    vLLM to batch-process all pages in the chunk together. This dramatically improves
    GPU utilization compared to single-page processing.

    When file_hash and fingerprint are provided, completed pages are checkpointed
    under DATA_DIR and skipped on subsequent runs (resume after interrupt).

    Args:
        file_content: Raw bytes of the PDF file
        filename: Original filename
        progress_callback: Optional callback function(completed_pages, total_pages)
        cancel_event: Optional threading.Event to signal cancellation
        max_workers: Maximum parallel chunk workers (default: MAX_PARALLEL_PAGES)
        pages_per_chunk: Pages per chunk for GPU batching (default: PAGES_PER_CHUNK)
        file_hash: SHA-256 of file content, used as the cache key
        fingerprint: Options fingerprint, used as the cache key
        **options: Processing options passed to process_document

    Returns:
        Combined API response dictionary with all pages

    Raises:
        CancellationError: If processing was cancelled by user
    """
    # Use defaults if not specified
    if max_workers is None:
        max_workers = MAX_PARALLEL_PAGES
    if pages_per_chunk is None:
        pages_per_chunk = PAGES_PER_CHUNK

    # Get total page count first
    total_pages = get_pdf_page_count(file_content)
    if total_pages == 0:
        raise RuntimeError("PDF has no pages or could not be read")

    # Resume from per-page checkpoints (source of truth is pages/*.json)
    cached_pages: dict[int, dict] = {}
    if file_hash and fingerprint:
        cached_pages = {
            page_num: result
            for page_num, result in load_completed_pages(file_hash, fingerprint).items()
            if 0 <= page_num < total_pages
        }

    results_dict: dict[int, dict] = dict(cached_pages)
    missing_pages = [i for i in range(total_pages) if i not in results_dict]

    if progress_callback:
        progress_callback(len(results_dict), total_pages)

    if not missing_pages:
        all_parsing_results = [results_dict[p] for p in range(total_pages)]
        return {
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {"layoutParsingResults": all_parsing_results},
        }

    if file_hash and fingerprint:
        write_partial_meta(
            file_hash, fingerprint, filename, total_pages, options
        )

    # Evenly distribute remaining pages across chunks (same idea as a full run)
    n_missing = len(missing_pages)
    num_chunks = (n_missing + pages_per_chunk - 1) // pages_per_chunk
    if num_chunks > 1:
        adjusted_chunk_size = (n_missing + num_chunks - 1) // num_chunks
        pages_per_chunk = max(1, adjusted_chunk_size)

    process_args = []
    for chunk_idx, offset in enumerate(range(0, n_missing, pages_per_chunk)):
        page_indices = missing_pages[offset : offset + pages_per_chunk]
        chunk_bytes = extract_pdf_pages(file_content, page_indices)
        process_args.append(
            (
                chunk_idx,
                page_indices,
                chunk_bytes,
                filename,
                options,
                cancel_event,
                file_hash,
                fingerprint,
            )
        )

    errors = []
    cancelled = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, args): args[1]  # page_indices
            for args in process_args
        }

        for future in as_completed(future_to_chunk):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                for f in future_to_chunk:
                    f.cancel()
                break

            chunk_idx, page_indices, parsing_results, error = future.result()
            page_label = ", ".join(str(p + 1) for p in page_indices)

            if error:
                if error == "Cancelled":
                    cancelled = True
                else:
                    errors.append(
                        f"Chunk {chunk_idx + 1} (pages {page_label}): {error}"
                    )
            elif parsing_results:
                for page_num, page_result in zip(page_indices, parsing_results):
                    results_dict[page_num] = page_result

            if progress_callback:
                progress_callback(len(results_dict), total_pages)

    if cancelled:
        raise CancellationError(
            f"Processing cancelled after {len(results_dict)} of {total_pages} pages"
        )

    if errors:
        error_summary = "; ".join(errors[:5])
        if len(errors) > 5:
            error_summary += f" ... and {len(errors) - 5} more errors"
        raise RuntimeError(f"Some chunks failed to process: {error_summary}")

    missing_after = [i + 1 for i in range(total_pages) if i not in results_dict]
    if missing_after:
        raise RuntimeError(
            f"Missing OCR results for page(s) {missing_after[:10]}"
            + (" ..." if len(missing_after) > 10 else "")
        )

    all_parsing_results = [results_dict[p] for p in range(total_pages)]
    all_parsing_results = restructure_parsing_results(
        all_parsing_results,
        prettify_markdown=bool(options.get("prettify_markdown", True)),
    )
    return {
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {"layoutParsingResults": all_parsing_results},
    }


# PaddleOCR-VL emits KaTeX that Typora and Streamlit both reject:
#   - inline `$ x $` (space after opening / before closing `$`)
#   - display `$$...$$` on one line (Typora only accepts a fenced block)
#   - a bare currency `$` (must be `\$` or KaTeX treats it as math)
# Normalize once at extract/load so preview, .md download, and ZIP stay in sync.
_CODE_SEGMENT = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)")
_MATH_HINT = re.compile(
    r"\\[a-zA-Z]+|[_^{}]|[=<>≤≥±∞∑∫√≠≈·×÷]|[+\-*/]"
)
_MATH_IDENT = re.compile(r"^[A-Za-z]\d{0,3}(\([^()]{0,24}\))?$")
_MATH_NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def _looks_like_math(inner: str) -> bool:
    """True when a `$...$` body is LaTeX / ident / number, not `$ or $` prose."""
    if _MATH_HINT.search(inner):
        return True
    return bool(_MATH_IDENT.fullmatch(inner) or _MATH_NUMBER.fullmatch(inner))


def _last_emitted_char(chunks: list[str]) -> str:
    """Last character already written, or newline if the buffer is empty."""
    if not chunks:
        return "\n"
    tail = chunks[-1]
    return tail[-1] if tail else "\n"


def _normalize_math_in_text(segment: str) -> str:
    """Rewrite math delimiters and escape leftover `$` as `\\$` (currency)."""
    n = len(segment)
    i = 0
    out: list[str] = []

    while i < n:
        j = i
        while j < n and segment[j] not in "$\\":
            j += 1
        if j > i:
            out.append(segment[i:j])
            i = j
            if i >= n:
                break

        # Already-escaped literal dollar — leave as `\$` (do not wrap as math)
        if segment[i] == "\\":
            if i + 1 < n and segment[i + 1] == "$":
                out.append("\\$")
                i += 2
                continue
            out.append("\\")
            i += 1
            continue

        # Display math: $$ ... $$ (possibly already a Typora block)
        if i + 1 < n and segment[i + 1] == "$":
            close = segment.find("$$", i + 2)
            if close != -1:
                inner = segment[i + 2 : close].strip()
                if inner:
                    if _last_emitted_char(out) != "\n":
                        out.append("\n")
                    out.append(f"$$\n{inner}\n$$")
                    i = close + 2
                    if i < n and segment[i] != "\n":
                        out.append("\n")
                    continue

        # Inline math `$...$` on one line, otherwise currency
        close = None
        k = i + 1
        while k < n:
            ch = segment[k]
            if ch == "\n":
                break
            if ch == "\\" and k + 1 < n:
                k += 2
                continue
            if ch == "$":
                if k + 1 < n and segment[k + 1] == "$":
                    break
                close = k
                break
            k += 1

        if close is not None:
            inner = segment[i + 1 : close].strip()
            if inner and _looks_like_math(inner):
                out.append(f"${inner}$")
                i = close + 1
                continue

        out.append("\\$")
        i += 1

    return "".join(out)


def normalize_markdown_math(text: str) -> str:
    """Make OCR markdown math compatible with Typora and Streamlit KaTeX.

    Idempotent. Fenced/inline code is left untouched. Display `$$...$$` is
    rewritten as a block; spaced `$ ... $` becomes `$...$` when the body looks
    like math. Any remaining `$` is currency and is escaped as `\\$`.
    """
    if not text or "$" not in text:
        return text

    pieces = _CODE_SEGMENT.split(text)
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith(("```", "~~~")) or (
            piece.startswith("`") and piece.endswith("`") and "\n" not in piece
        ):
            out.append(piece)
        else:
            out.append(_normalize_math_in_text(piece))
    return "".join(out)


def extract_markdown_from_response(api_response: dict, base_filename: str = "document") -> tuple[str, dict]:
    """
    Extract markdown text and images from API response.

    Rewrites image paths in the markdown to match the ZIP archive structure:
    - Original: imgs/img_xxx.jpg
    - Rewritten: {base_filename}_images/page_{n}/img_xxx.jpg (multi-page)
    - Rewritten: {base_filename}_images/img_xxx.jpg (single-page)

    Args:
        api_response: The API response dictionary
        base_filename: Base filename for constructing image paths (without extension)

    Returns:
        Tuple of (markdown_text, images_dict)
    """
    result = api_response.get("result", {})
    parsing_results = result.get("layoutParsingResults", [])

    if not parsing_results:
        return "# No content detected", {}

    markdown_parts = []
    all_images = {}
    images_dir = f"{base_filename}_images"

    for i, page_result in enumerate(parsing_results):
        markdown_info = page_result.get("markdown", {})
        markdown_text = markdown_info.get("text", "")
        images = markdown_info.get("images", {})

        # Rewrite image paths in markdown to match ZIP structure
        for original_path, img_data in images.items():
            # Extract just the filename from the original path (e.g., "imgs/img_xxx.jpg" -> "img_xxx.jpg")
            img_filename = Path(original_path).name

            if len(parsing_results) > 1:
                # Multi-page: store under page subdirectory
                new_path = f"{images_dir}/page_{i + 1}/{img_filename}"
                all_images[f"page_{i + 1}/{img_filename}"] = img_data
            else:
                # Single-page: store directly in images directory
                new_path = f"{images_dir}/{img_filename}"
                all_images[img_filename] = img_data

            # Replace original path with new path in markdown
            markdown_text = markdown_text.replace(original_path, new_path)

        if len(parsing_results) > 1:
            markdown_parts.append(f"<!-- Page {i + 1} -->\n\n{markdown_text}")
        else:
            markdown_parts.append(markdown_text)

    full_markdown = "\n\n---\n\n".join(markdown_parts)
    return normalize_markdown_math(full_markdown), all_images


def create_download_zip(markdown_text: str, images: dict, base_filename: str) -> bytes:
    """
    Create a ZIP file containing the markdown and associated images.

    The images dict keys should already contain the relative paths that match
    the image references in the markdown (e.g., "page_1/img_xxx.jpg").
    Images will be stored under {base_filename}_images/ directory.

    Returns:
        Bytes of the ZIP file
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add markdown file
        zip_file.writestr(f"{base_filename}.md", markdown_text.encode("utf-8"))

        # Add images - paths in images dict are relative (e.g., "page_1/img.jpg")
        # Store them under {base_filename}_images/ to match markdown references
        images_dir = f"{base_filename}_images"
        for img_path, img_data in images.items():
            img_bytes = decode_base64_image(img_data)
            zip_file.writestr(f"{images_dir}/{img_path}", img_bytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def display_processing_options() -> dict:
    """Display and collect processing options from sidebar."""
    st.sidebar.header("⚙️ Processing Options")
    if OCR_QUALITY_FIRST:
        st.sidebar.success(
            "Quality-first pipeline is on (`OCR_QUALITY_FIRST=true`). "
            "Seal OCR, figure-text OCR, high-recall layout, and cross-page "
            "reconstruction are enabled. Recreate the API container after "
            "changing this variable."
        )
    else:
        st.sidebar.caption(
            "Speed-first defaults. Set `OCR_QUALITY_FIRST=true` in `.env` "
            "and recreate the API container for higher recall."
        )

    # Widget keys include the mode so flipping the env var does not keep
    # stale Streamlit checkbox state from the other mode.
    mode_key = "q" if OCR_QUALITY_FIRST else "s"

    options = {
        "use_doc_orientation_classify": st.sidebar.checkbox(
            "Document Orientation Classification",
            value=True if OCR_QUALITY_FIRST else USE_DOC_ORIENTATION_CLASSIFY,
            key=f"opt_orient_{mode_key}",
            help="Automatically detect and correct document orientation",
        ),
        "use_doc_unwarping": st.sidebar.checkbox(
            "Document Unwarping",
            value=True if OCR_QUALITY_FIRST else USE_DOC_UNWARPING,
            key=f"opt_unwarp_{mode_key}",
            help="Correct curved or warped document images",
        ),
        "use_layout_detection": st.sidebar.checkbox(
            "Layout Detection",
            value=True if OCR_QUALITY_FIRST else USE_LAYOUT_DETECTION,
            key=f"opt_layout_{mode_key}",
            help="Detect document layout structure (recommended)",
        ),
        "use_chart_recognition": st.sidebar.checkbox(
            "Chart Recognition",
            value=True if OCR_QUALITY_FIRST else USE_CHART_RECOGNITION,
            key=f"opt_chart_{mode_key}",
            help="Enable chart and diagram recognition",
        ),
        "prettify_markdown": st.sidebar.checkbox(
            "Prettify Markdown",
            value=PRETTIFY_MARKDOWN,
            help="Format markdown output for better readability",
        ),
        "visualize": st.sidebar.checkbox(
            "Show Visualization",
            value=VISUALIZE_RESULTS,
            help="Display intermediate processing results (slower)",
        ),
    }

    return options


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"📄 {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)

    # Check API health
    with st.spinner("Checking API service status..."):
        api_healthy = check_api_health()

    if not api_healthy:
        st.error(
            "⚠️ PaddleOCR-VL API service is not available. "
            "Please ensure the service is running."
        )
        st.info(
            f"Expected API URL: `{PADDLEOCR_VL_API_URL}`\n\n"
            "If running locally, start the service with:\n"
            "```bash\ndocker compose up -d\n```"
        )
        return

    st.success("✅ PaddleOCR-VL API service is healthy")

    # Processing options
    options = display_processing_options()
    # Fingerprint of the current options; identical file + options share one cache entry
    options_fp = options_fingerprint(options)

    # File upload section
    st.header("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or image files",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp"],
        accept_multiple_files=True,
        help=f"Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP. Max size: {MAX_FILE_SIZE_MB}MB",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more documents to get started")
        return

    # Initialize session state
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = {}
    if "files_to_process" not in st.session_state:
        st.session_state.files_to_process = {}
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Store uploaded files content for processing
    valid_files = []
    for uploaded_file in uploaded_files:
        is_valid, validation_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {uploaded_file.name}: {validation_msg}")
            continue

        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        file_hash = compute_sha256(file_content)
        valid_files.append((uploaded_file, file_content, file_hash))

    if not valid_files:
        return

    # Preview section
    st.header("👁️ Document Preview")
    st.caption("Review your documents before processing")

    for uploaded_file, file_content, file_hash in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            display_file_preview(uploaded_file, file_content)

            # Store file content for later processing
            st.session_state.files_to_process[file_key] = {
                "name": uploaded_file.name,
                "content": file_content,
                "hash": file_hash,
                "fingerprint": options_fp,
            }

    # Start OCR button
    st.header("🚀 Start Processing")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_button = st.button(
            "🔍 Start OCR",
            type="primary",
            disabled=st.session_state.is_processing,
            help="Process all uploaded documents with OCR",
        )
    with col2:
        cancel_button = st.button(
            "⏹️ Cancel",
            type="secondary",
            disabled=not st.session_state.is_processing,
            help="Cancel the current processing",
        )
    with col3:
        st.caption(f"Ready to process {len(valid_files)} document(s)")

    # Handle cancel button
    if cancel_button:
        st.session_state.cancel_requested = True
        st.warning("⏹️ Cancellation requested... waiting for current pages to finish.")

    if not start_button:
        st.info("👆 Click 'Start OCR' to begin processing your documents")
        return

    # Reset cancel flag when starting new processing
    st.session_state.cancel_requested = False
    st.session_state.is_processing = True

    # Processing section
    st.header("🔄 Processing Results")

    # Create a cancel event for thread communication
    cancel_event = threading.Event()

    for uploaded_file, file_content, file_hash in valid_files:
        # Check for cancellation between files
        if st.session_state.cancel_requested:
            cancel_event.set()
            st.warning("⏹️ Skipping remaining files due to cancellation.")
            break

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        is_pdf = uploaded_file.name.lower().endswith(".pdf")

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            # Session cache is keyed by name+size; only reuse when options match.
            # A fingerprint mismatch falls through to disk cache / reprocess.
            cached = st.session_state.processing_results.get(file_key)
            if cached is not None and cached.get("fingerprint") == options_fp:
                if cached.get("from_disk_cache"):
                    st.info(
                        "📋 Using results restored from the disk cache (persisted from a previous run). "
                        "Change the processing options to reprocess."
                    )
                else:
                    st.info("📋 Using cached results. Re-upload to reprocess.")
                markdown_text = normalize_markdown_math(cached["markdown"])
                cached["markdown"] = markdown_text
                images = cached["images"]
            elif (disk_hit := load_cached_result(file_hash, options_fp)) is not None:
                # Stored on disk from an earlier run - serve instantly, no API call
                st.session_state.processing_results[file_key] = disk_hit
                st.success(
                    "⚡ Instant from disk cache — file previously processed with these options"
                )
                markdown_text = disk_hit["markdown"]
                images = disk_hit["images"]
            else:
                try:
                    start_time = time.time()
                    page_count = 0

                    if is_pdf:
                        # Use batch processing for PDFs to handle all pages
                        page_count = get_pdf_page_count(file_content)
                        cached_pages = load_completed_pages(file_hash, options_fp)
                        already_done = sum(
                            1 for p in cached_pages if 0 <= p < page_count
                        )

                        if already_done and already_done < page_count:
                            st.info(
                                f"♻️ Resuming: {already_done}/{page_count} pages already "
                                "cached. Only remaining pages will be sent to the API."
                            )
                        elif already_done == page_count and page_count > 0:
                            st.info(
                                f"♻️ Assembling {page_count} cached pages into the final result."
                            )

                        # Create containers for progress and cancel status
                        progress_container = st.empty()
                        initial_ratio = (
                            already_done / page_count if page_count else 0.0
                        )
                        progress_bar = progress_container.progress(
                            min(1.0, max(0.0, initial_ratio)),
                            text=(
                                f"Processing page {already_done}/{page_count}... "
                                "(Cancel with button above)"
                            ),
                        )

                        def update_progress(current, total):
                            # Check cancel state from session
                            if st.session_state.cancel_requested:
                                cancel_event.set()
                            ratio = current / total if total else 0
                            progress_bar.progress(
                                min(1.0, max(0.0, ratio)),
                                text=f"Processing page {current}/{total}...",
                            )

                        api_response = process_pdf_in_batches(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            progress_callback=update_progress,
                            cancel_event=cancel_event,
                            file_hash=file_hash,
                            fingerprint=options_fp,
                            **options,
                        )
                        progress_container.empty()
                    else:
                        # Process single image directly
                        page_count = 1
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            api_response = process_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                **options,
                            )

                    # Extract markdown with image paths rewritten to match ZIP structure
                    base_filename = display_stem(uploaded_file.name)
                    markdown_text, images = extract_markdown_from_response(api_response, base_filename)

                    processing_time = time.time() - start_time
                    st.success(f"✅ Processed in {processing_time:.1f} seconds")

                    # Cache results (fingerprint so option changes miss this entry)
                    st.session_state.processing_results[file_key] = {
                        "markdown": markdown_text,
                        "images": images,
                        "response": api_response,
                        "display_name": uploaded_file.name,
                        "fingerprint": options_fp,
                    }

                    # Persist to disk for instant serving on re-upload (never raises)
                    save_result_to_disk(
                        hash_str=file_hash,
                        fingerprint=options_fp,
                        display_name=uploaded_file.name,
                        markdown_text=markdown_text,
                        images=images,
                        page_count=page_count,
                        options=options,
                    )

                except requests.Timeout:
                    st.error(
                        f"⏱️ Request timed out after {API_TIMEOUT} seconds. "
                        "Try processing smaller files or increase timeout."
                    )
                    continue
                except requests.RequestException as e:
                    st.error(f"🌐 Network error: {str(e)}")
                    continue
                except CancellationError as e:
                    st.warning(f"⏹️ {str(e)}")
                    if is_pdf:
                        st.info(
                            "Completed pages were saved to disk. Click Start OCR again "
                            "with the same file and options to resume."
                        )
                    st.session_state.is_processing = False
                    st.stop()  # Stop further processing
                except RuntimeError as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    if is_pdf and "no pages" not in str(e).lower():
                        st.info(
                            "Successfully processed pages were saved. Click Start OCR again "
                            "with the same file and options to resume."
                        )
                    continue
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    continue

            # Display results in tabs
            tab_preview, tab_raw, tab_download = st.tabs(
                ["📖 Preview", "📝 Raw Markdown", "💾 Download"]
            )

            with tab_preview:
                # Count pages in markdown (separated by ---)
                page_separators = markdown_text.count("\n\n---\n\n")
                total_pages = page_separators + 1 if page_separators > 0 else 1

                if total_pages > MAX_PREVIEW_PAGES:
                    # Split by page separator and show only first N pages
                    pages = markdown_text.split("\n\n---\n\n")
                    truncated_md = "\n\n---\n\n".join(pages[:MAX_PREVIEW_PAGES])

                    st.warning(
                        f"⚠️ Showing preview of first {MAX_PREVIEW_PAGES} pages only "
                        f"(document has {total_pages} pages). Use 'Raw Markdown' tab or download for full content."
                    )
                    st.markdown(truncated_md)

                    # Optional: expandable full preview
                    with st.expander(f"📄 Show all {total_pages} pages (may be slow)"):
                        st.markdown(markdown_text)
                else:
                    st.markdown(markdown_text)

                # Display embedded images if any (also limit images shown)
                if images:
                    st.subheader("🖼️ Extracted Images")
                    max_images_preview = MAX_PREVIEW_PAGES * 3  # ~3 images per page max
                    images_list = list(images.items())
                    display_images = images_list[:max_images_preview]

                    cols = st.columns(min(len(display_images), 3))
                    for idx, (img_path, img_data) in enumerate(display_images):
                        with cols[idx % 3]:
                            img_bytes = decode_base64_image(img_data)
                            st.image(img_bytes, caption=img_path, width="stretch")

                    if len(images_list) > max_images_preview:
                        st.info(
                            f"📷 Showing {max_images_preview} of {len(images_list)} images. "
                            "Download ZIP for all images."
                        )

            with tab_raw:
                st.code(markdown_text, language="markdown")

            with tab_download:
                # Download names come from the display name stored with the
                # result (the original upload name for disk-cache hits)
                base_filename = display_stem(
                    st.session_state.processing_results[file_key].get(
                        "display_name", uploaded_file.name
                    )
                )

                col1, col2 = st.columns(2)

                with col1:
                    # Download markdown only
                    st.download_button(
                        label="📄 Download Markdown (.md)",
                        data=markdown_text.encode("utf-8"),
                        file_name=f"{base_filename}.md",
                        mime="text/markdown",
                    )

                with col2:
                    # Download as ZIP with images
                    if images:
                        zip_data = create_download_zip(
                            markdown_text, images, base_filename
                        )
                        st.download_button(
                            label="📦 Download ZIP (with images)",
                            data=zip_data,
                            file_name=f"{base_filename}_ocr_result.zip",
                            mime="application/zip",
                        )
                    else:
                        st.info("No images to include in ZIP")

            # Display visualization if enabled and available
            if options["visualize"] and file_key in st.session_state.processing_results:
                cached = st.session_state.processing_results[file_key]
                if cached.get("from_disk_cache"):
                    st.caption(
                        "🔍 Visualization images are not stored in the disk cache. "
                        "Reprocess the file to view them."
                    )
                else:
                    response = cached.get("response", {})
                    result = response.get("result", {})
                    parsing_results = result.get("layoutParsingResults", [])

                    for page_result in parsing_results:
                        output_images = page_result.get("outputImages", {})
                        if output_images:
                            st.subheader("🔍 Processing Visualization")
                            vis_cols = st.columns(min(len(output_images), 2))
                            for idx, (img_name, img_data) in enumerate(
                                output_images.items()
                            ):
                                if img_data:
                                    with vis_cols[idx % 2]:
                                        img_bytes = decode_base64_image(img_data)
                                        st.image(
                                            img_bytes,
                                            caption=img_name.replace("_", " ").title(),
                                            width="stretch",
                                        )

    # Batch download section
    if len(st.session_state.processing_results) > 1:
        st.header("📦 Batch Download")
        if st.button("Download All Results as ZIP"):
            batch_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                batch_zip_buffer, "w", zipfile.ZIP_DEFLATED
            ) as batch_zip:
                for file_key, data in st.session_state.processing_results.items():
                    base_name = display_stem(
                        data.get("display_name", file_key.rsplit("_", 1)[0])
                    )
                    # Store each document in its own directory
                    batch_zip.writestr(
                        f"{base_name}/{base_name}.md", data["markdown"].encode("utf-8")
                    )
                    # Images are stored to match markdown references: {base_name}_images/...
                    images_dir = f"{base_name}_images"
                    for img_path, img_data in data["images"].items():
                        img_bytes = decode_base64_image(img_data)
                        batch_zip.writestr(f"{base_name}/{images_dir}/{img_path}", img_bytes)

            batch_zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Results",
                data=batch_zip_buffer.getvalue(),
                file_name="all_ocr_results.zip",
                mime="application/zip",
            )

    # Reset processing state
    st.session_state.is_processing = False
    st.session_state.cancel_requested = False

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io) and "
        "[PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) using vLLM backend"
    )


if __name__ == "__main__":
    main()
