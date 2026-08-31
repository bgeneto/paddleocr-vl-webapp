"""
PaddleOCR-VL Document Parser Streamlit Application

This application provides a web interface for document OCR using PaddleOCR-VL
(llama.cpp by default; vLLM via compose.vllm.yaml).
"""

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
    "Upload PDF or image files to convert them to Markdown using PaddleOCR-VL",
)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "99"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "250"))
# Parallel workers - keep low since API may serialize requests anyway
MAX_PARALLEL_PAGES = int(os.getenv("MAX_PARALLEL_PAGES", "8"))
MAX_PREVIEW_PAGES = int(os.getenv("MAX_PREVIEW_PAGES", "10"))  # Limit preview rendering
# Pages per chunk - HIGHER = better GPU batching (vLLM processes all pages in chunk together)
# This is the KEY setting for GPU utilization. Increase if you have enough VRAM.
PAGES_PER_CHUNK = int(
    os.getenv("PAGES_PER_CHUNK", "16")
)  # Pages per API request for GPU batching

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
# HTML comments like <!-- Page 2 --> between pages (RAG chunking). On by
# default. When on, quality-first does not concatenate pages into one blob.
INCLUDE_PAGE_MARKERS = os.getenv("INCLUDE_PAGE_MARKERS", "true").lower() == "true"
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
# file content and a fingerprint of the processing options:
#   DATA_DIR/<sha256>/<options-fingerprint>/{meta.json,result.md,result.zip,pages/}
# Re-uploading a byte-identical file with the same options is served instantly
# from disk (no API call). In-progress PDF jobs also checkpoint each completed
# page under pages/{n:04d}.json so a later Start OCR can skip already-OCR'd
# pages. The directory survives container recreation via the ./data bind mount
# (see compose.yaml). Entries older than DATA_CACHE_RETENTION_DAYS are removed
# automatically (see schedule_cache_cleanup); this cache is not durable storage.
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data")))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
# 0 disables automatic eviction. Positive values are a maximum age in days.
DATA_CACHE_RETENTION_DAYS = int(os.getenv("DATA_CACHE_RETENTION_DAYS", "45"))

logger = logging.getLogger(__name__)

_SHA256_DIR_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_cleanup_lock = threading.Lock()
_last_cache_cleanup_mono: float = 0.0
_cleanup_thread: threading.Thread | None = None
# Retention is measured in days; do not walk DATA_DIR on every Streamlit rerun.
_CACHE_CLEANUP_MIN_INTERVAL_SEC = 60 * 60

# (option key, short name) pairs in fixed order; defines the fingerprint format
_OPTION_FINGERPRINT_KEYS = (
    ("use_doc_orientation_classify", "orient"),
    ("use_doc_unwarping", "unwarp"),
    ("use_layout_detection", "layout"),
    ("use_chart_recognition", "chart"),
    ("prettify_markdown", "pretty"),
    ("visualize", "vis"),
    ("include_page_markers", "pages"),
)
# Keys accepted by process_document / the layout-parsing request body.
_LAYOUT_API_OPTION_KEYS = tuple(
    key for key, _ in _OPTION_FINGERPRINT_KEYS if key != "include_page_markers"
)


def _layout_api_options(options: dict) -> dict:
    """Sidebar keys that belong in the layout-parsing request body."""
    return {key: options[key] for key in _LAYOUT_API_OPTION_KEYS if key in options}


def compute_sha256(content: bytes) -> str:
    """Compute the SHA-256 hex digest of file content (already fully in memory)."""
    return hashlib.sha256(content).hexdigest()


def options_fingerprint(options: dict) -> str:
    """Deterministic compact fingerprint of the processing options.

    Example: 'orient=1_unwarp=0_layout=1_chart=0_pretty=1_vis=0_pages=1'.
    Key order is fixed here, so the ordering of the caller's dict does not
    matter.
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


def _dir_is_writable(path: Path) -> bool:
    """Return True if path exists (or can be created) and a file can be written."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _parse_iso_datetime(value: object) -> datetime | None:
    """Parse a meta.json ISO timestamp into an aware UTC datetime."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cache_entry_timestamp(entry_dir: Path) -> datetime | None:
    """Best-effort last-write time for a fingerprint cache directory.

    Prefers meta.json updated_at/created_at, then directory mtimes (the
    fingerprint folder and pages/ checkpoints). Returns None if nothing
    can be stat'd so the caller will not delete the entry.
    """
    candidates: list[datetime] = []
    meta_path = entry_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            meta = None
        if isinstance(meta, dict):
            for key in ("updated_at", "created_at"):
                parsed = _parse_iso_datetime(meta.get(key))
                if parsed is not None:
                    candidates.append(parsed)
    for path in (entry_dir, entry_dir / "pages"):
        try:
            if path.exists() and not path.is_symlink():
                candidates.append(
                    datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                )
        except OSError:
            pass
    if not candidates:
        return None
    return max(candidates)


def _is_sha256_cache_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.is_symlink()
        and _SHA256_DIR_RE.fullmatch(path.name) is not None
    )


def cleanup_old_cache_dirs() -> int:
    """Delete DATA_DIR fingerprint folders older than DATA_CACHE_RETENTION_DAYS.

    Must not be called on the Streamlit script thread: listing and rmtree can
    take noticeable time. Use schedule_cache_cleanup() from main() instead.
    Does not touch Streamlit session state. Empty SHA-256 parent dirs are
    removed after their fingerprints. Never raises.

    Returns the number of fingerprint directories removed.
    """
    if DATA_CACHE_RETENTION_DAYS < 1:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=DATA_CACHE_RETENTION_DAYS)
    try:
        data_root = DATA_DIR.resolve()
        children = list(DATA_DIR.iterdir())
    except OSError as e:
        logger.warning("Cache cleanup could not list %s: %s", DATA_DIR, e)
        return 0

    removed = 0
    try:
        for hash_dir in children:
            if not _is_sha256_cache_dir(hash_dir):
                continue
            try:
                if not hash_dir.resolve().is_relative_to(data_root):
                    continue
            except OSError:
                continue
            try:
                fingerprints = list(hash_dir.iterdir())
            except OSError as e:
                logger.warning("Cache cleanup could not list %s: %s", hash_dir, e)
                continue
            for entry_dir in fingerprints:
                if not entry_dir.is_dir() or entry_dir.is_symlink():
                    continue
                try:
                    if not entry_dir.resolve().is_relative_to(data_root):
                        continue
                except OSError:
                    continue
                stamp = _cache_entry_timestamp(entry_dir)
                if stamp is None or stamp > cutoff:
                    continue
                try:
                    shutil.rmtree(entry_dir)
                    removed += 1
                    logger.info(
                        "Removed expired cache entry %s (older than %s days)",
                        entry_dir,
                        DATA_CACHE_RETENTION_DAYS,
                    )
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning(
                        "Failed to remove expired cache entry %s: %s", entry_dir, e
                    )
            try:
                remaining = any(hash_dir.iterdir())
            except OSError:
                remaining = True
            if not remaining:
                try:
                    hash_dir.rmdir()
                except OSError:
                    pass
    except Exception as e:
        logger.warning("Cache cleanup failed: %s", e)
        return removed

    if removed:
        logger.info("Cache cleanup removed %s expired folder(s)", removed)
    return removed


def _cache_cleanup_worker() -> None:
    """Background target: run disk cleanup and swallow unexpected errors."""
    try:
        cleanup_old_cache_dirs()
    except Exception as e:
        logger.warning("Cache cleanup worker failed: %s", e)


def schedule_cache_cleanup() -> None:
    """Start cache eviction on a daemon thread; return without waiting.

    Safe to call from every Streamlit rerun. At most one worker runs at a
    time, and a new one is not started more than once per
    _CACHE_CLEANUP_MIN_INTERVAL_SEC.
    """
    global _last_cache_cleanup_mono, _cleanup_thread

    if DATA_CACHE_RETENTION_DAYS < 1:
        return

    with _cleanup_lock:
        now_mono = time.monotonic()
        if (
            _last_cache_cleanup_mono > 0
            and now_mono - _last_cache_cleanup_mono < _CACHE_CLEANUP_MIN_INTERVAL_SEC
        ):
            return
        if _cleanup_thread is not None and _cleanup_thread.is_alive():
            return
        # Claim the interval before start() so later reruns do not spawn more
        # threads while this one is still walking the disk.
        _last_cache_cleanup_mono = now_mono
        _cleanup_thread = threading.Thread(
            target=_cache_cleanup_worker,
            name="cache-cleanup",
            daemon=True,
        )
        _cleanup_thread.start()


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
) -> bool:
    """Persist a successful result so re-uploads can be served instantly.

    Never raises: a storage failure must not break a successful in-memory
    result (logged; the worker surfaces a warning). All files are written
    atomically (temp file + rename). Returns True on success, False on failure.
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
            "options": {
                key: bool(options.get(key)) for key, _ in _OPTION_FINGERPRINT_KEYS
            },
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
        return True
    except Exception as e:
        logger.warning(
            "Failed to save result to disk cache (%s/%s): %s", hash_str, fingerprint, e
        )
        return False


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
                    images[name[len(images_prefix) :]] = base64.b64encode(data).decode(
                        "ascii"
                    )

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
            "options": {
                key: bool(options.get(key)) for key, _ in _OPTION_FINGERPRINT_KEYS
            },
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


class OcrJob:
    """Thread-safe OCR job so Streamlit can keep Cancel clickable while workers run.

    Streamlit will not handle widget clicks while the script is blocked in an
    HTTP / thread-pool wait. The worker updates this object; the script only
    reads snapshots and sets ``cancel_event``.
    """

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._lock = threading.RLock()
        self.status = "running"  # running | cancelling | done | cancelled | error
        self.current_file = ""
        self.progress = (0, 0)
        self.detail = ""
        self.bar_ratio: float | None = None
        self.results: dict[str, dict] = {}
        self.messages: list[tuple[str, str]] = []
        self.error: str | None = None

    def request_cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            if self.status == "running":
                self.status = "cancelling"

    def set_current_file(self, name: str) -> None:
        with self._lock:
            self.current_file = name

    def set_progress(
        self,
        current: int,
        total: int,
        detail: str | None = None,
        bar_ratio: float | None = None,
    ) -> None:
        with self._lock:
            self.progress = (int(current), int(total))
            if detail is not None:
                self.detail = detail
            self.bar_ratio = bar_ratio

    def add_message(self, level: str, text: str) -> None:
        with self._lock:
            self.messages.append((level, text))

    def store_result(self, file_key: str, result: dict) -> None:
        with self._lock:
            self.results[file_key] = result

    def finish(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error:
                self.error = error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "current_file": self.current_file,
                "progress": self.progress,
                "detail": self.detail,
                "bar_ratio": self.bar_ratio,
                "results": dict(self.results),
                "messages": list(self.messages),
                "error": self.error,
            }


def _on_start_ocr() -> None:
    """Runs before widgets render so Cancel is enabled on the Start-OCR rerun."""
    st.session_state.is_processing = True
    st.session_state.start_ocr_requested = True


def _on_cancel_ocr() -> None:
    """Runs before widgets render so the worker sees cancel on this rerun."""
    job = st.session_state.get("ocr_job")
    if job is not None:
        job.request_cancel()


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
            # Serving schema types this as Tuple, then the validator rejects
            # tuples. A JSON array therefore 400s. A single number is accepted.
            "layoutUnclipRatio": 1.1,
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
    parsing_results: list,
    prettify_markdown: bool = True,
    concatenate_pages: bool = True,
) -> list:
    """Merge tables / relevel titles / optionally concatenate pages after OCR.

    No-ops (returns the input list) unless OCR_QUALITY_FIRST is on and there
    are at least two pages with prunedResult. Failures are logged and the
    original per-page results are kept.

    concatenate_pages must be False when HTML page comments are wanted:
    extract_markdown_from_response only emits <!-- Page N --> when more than
    one layoutParsingResults item remains.
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
                "concatenatePages": concatenate_pages,
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


# llama.cpp / vLLM Prometheus counters (any one present is enough).
_VLM_METRIC_NAMES = (
    "llamacpp:tokens_predicted",
    "llamacpp_tokens_predicted",
    "llamacpp:n_tokens_predicted",
    "llamacpp:predicted_tokens_total",
    "vllm:generation_tokens_total",
)
_VLM_PROMPT_METRIC_NAMES = (
    "llamacpp:tokens_evaluated",
    "llamacpp_tokens_evaluated",
    "llamacpp:prompt_tokens_total",
    "vllm:prompt_tokens_total",
)
# Decode+prefill work units per PDF page (crops vary; this only paces the bar).
_VLM_TOKENS_PER_PAGE = 500.0
_VLM_METRICS_URL = (
    os.getenv("LLAMA_SERVER_URL", "http://paddleocr-vlm-server:8080").rstrip("/")
    + "/metrics"
)


def _prometheus_counter(text: str, names: tuple[str, ...]) -> float | None:
    """Return the first matching Prometheus counter value, or None."""
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        for name in names:
            if raw.startswith(name + " ") or raw.startswith(name + "{"):
                try:
                    return float(raw.rsplit(None, 1)[-1])
                except ValueError:
                    return None
    return None


def _vlm_work_units() -> float | None:
    """Monotonic VLM work counter from llama.cpp/vLLM /metrics, if reachable."""
    try:
        response = requests.get(_VLM_METRICS_URL, timeout=0.35)
        if response.status_code != 200:
            return None
        body = response.text
        predicted = _prometheus_counter(body, _VLM_METRIC_NAMES)
        prompt = _prometheus_counter(body, _VLM_PROMPT_METRIC_NAMES)
        if predicted is None and prompt is None:
            return None
        return (predicted or 0.0) + (prompt or 0.0) / 8.0
    except requests.RequestException:
        return None


def _page_range_label(page_indices: list[int], total_pages: int) -> str:
    first = page_indices[0] + 1
    last = page_indices[-1] + 1
    if first == last:
        return f"page {first} of {total_pages}"
    return f"pages {first}–{last} of {total_pages}"


class _ChunkProgress:
    """Thread-safe page progress across parallel in-flight API chunks.

    The page count is pages actually saved. In-flight work is one shared
    llama/vLLM /metrics + time estimate so fat chunks do not each claim
    ``n_pages - 1`` from the same global token counter and freeze at 98%.
    """

    def __init__(self, committed: int, total: int, callback) -> None:
        self._lock = threading.Lock()
        self.committed = committed
        self.total = total
        self.inflight: dict[int, int] = {}
        self.labels: dict[int, str] = {}
        self.callback = callback
        self._units0: float | None = None
        self._last_units: float | None = None
        self._t0: float | None = None
        self._peak_bar = float(committed)

    def begin_chunk(self, idx: int, n_pages: int, label: str) -> None:
        units = _vlm_work_units()
        with self._lock:
            if not self.inflight:
                self._units0 = units
                self._last_units = units
                self._t0 = time.perf_counter()
            elif units is not None:
                self._last_units = units
            self.inflight[idx] = n_pages
            self.labels[idx] = label
            self._emit_locked()

    def pulse(
        self,
        idx: int,
        range_label: str | None = None,
        *,
        verb_prefix: str | None = None,
    ) -> None:
        units = _vlm_work_units()
        with self._lock:
            if units is not None:
                self._last_units = units
            if idx in self.labels and range_label is not None:
                if verb_prefix is not None:
                    self.labels[idx] = f"{verb_prefix} {range_label}"
                else:
                    _extra, phase = self._estimate_locked()
                    verb = (
                        "recognizing"
                        if phase == "recognize"
                        else "detecting layout on"
                    )
                    self.labels[idx] = f"{verb} {range_label}"
            self._emit_locked()

    def finish_chunk(self, idx: int, n_pages: int) -> None:
        with self._lock:
            self.inflight.pop(idx, None)
            self.labels.pop(idx, None)
            self.committed += n_pages
            if not self.inflight:
                self._units0 = None
                self._t0 = None
                self._peak_bar = float(self.committed)
            self._emit_locked()

    def abandon_chunk(self, idx: int) -> None:
        with self._lock:
            self.inflight.pop(idx, None)
            self.labels.pop(idx, None)
            if not self.inflight:
                self._units0 = None
                self._t0 = None
            self._emit_locked()

    def _estimate_locked(self) -> tuple[float, str]:
        n_inflight = sum(self.inflight.values())
        if not n_inflight:
            return 0.0, "done"
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        units = self._last_units
        phase = "layout"
        expected_tok = max(_VLM_TOKENS_PER_PAGE * n_inflight, 1.0)
        expected_s = max(2.0 * n_inflight, 12.0)
        time_frac = 1.0 - math.exp(-elapsed / expected_s)
        if self._units0 is not None and units is not None:
            delta = max(0.0, units - self._units0)
            if delta >= 8:
                phase = "recognize"
            token_frac = 1.0 - math.exp(-delta / expected_tok) if delta else 0.0
            frac = max(token_frac, 0.35 * time_frac)
        else:
            frac = time_frac
        return n_inflight * 0.88 * frac, phase

    def _emit_locked(self) -> None:
        extra, _phase = (
            self._estimate_locked() if self.inflight else (0.0, "done")
        )
        saved = self.committed
        remaining = max(0, self.total - saved)
        if remaining:
            extra = min(extra, max(0.0, remaining - 0.2))
        else:
            extra = 0.0
        bar_n = min(float(self.total), max(self._peak_bar, saved + extra))
        if saved >= self.total and not self.inflight:
            bar_n = float(self.total)
        self._peak_bar = bar_n
        labels = [self.labels[k] for k in sorted(self.labels)]
        if labels:
            detail = "OCR " + "; ".join(labels)
        elif self.committed >= self.total and self.total:
            detail = ""
        else:
            detail = f"{saved}/{self.total} pages saved"
        bar_ratio = (bar_n / self.total) if self.total else 0.0
        if self.inflight:
            bar_ratio = min(bar_ratio, 0.97)
        if self.callback:
            self.callback(saved, self.total, detail, bar_ratio)


def process_chunk(args: tuple) -> tuple[int, list[int], list | None, str | None]:
    """
    Process a chunk of PDF pages. Helper function for parallel chunk processing.

    Each chunk contains multiple pages, which allows vLLM to batch-process them
    together for better GPU utilization. On success, each page is checkpointed
    to disk immediately (in this worker) so a crash of the Streamlit run still
    preserves completed pages.

    Args:
        args: Tuple of (chunk_index, page_indices, chunk_content, filename,
            options, cancel_event, file_hash, fingerprint, tracker, total_pages).
            page_indices are 0-indexed original PDF page numbers, in the same
            order as pages in chunk_content.

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
        tracker,
        total_pages,
    ) = args

    n_pages = len(page_indices)
    label = _page_range_label(page_indices, total_pages)

    # Check for cancellation before starting
    if cancel_event and cancel_event.is_set():
        return (chunk_index, page_indices, None, "Cancelled")

    if tracker is not None:
        tracker.begin_chunk(chunk_index, n_pages, f"{label} (starting)")

    box: dict = {}

    def _post() -> None:
        try:
            box["response"] = process_document(
                file_content=chunk_content,
                filename=filename,
                **_layout_api_options(options),
            )
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(
        target=_post, name=f"ocr-chunk-{chunk_index}", daemon=True
    )
    worker.start()

    while worker.is_alive():
        worker.join(0.35)
        if tracker is not None:
            tracker.pulse(chunk_index, label)

    if "error" in box:
        if tracker is not None:
            tracker.abandon_chunk(chunk_index)
        return (chunk_index, page_indices, None, str(box["error"]))
    if "response" not in box:
        if tracker is not None:
            tracker.abandon_chunk(chunk_index)
        return (chunk_index, page_indices, None, "Chunk request failed")

    try:
        chunk_response = box["response"]
        result = chunk_response.get("result", {})
        parsing_results = result.get("layoutParsingResults", [])

        if not parsing_results or len(parsing_results) != len(page_indices):
            got = 0 if not parsing_results else len(parsing_results)
            if tracker is not None:
                tracker.abandon_chunk(chunk_index)
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
                if tracker is not None:
                    tracker.pulse(chunk_index, label, verb_prefix="saving")

        if tracker is not None:
            tracker.finish_chunk(chunk_index, n_pages)

        return (chunk_index, page_indices, parsing_results, None)
    except Exception as e:
        if tracker is not None:
            tracker.abandon_chunk(chunk_index)
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
        progress_callback: Optional callback(saved_pages, total_pages, detail, bar_ratio)
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

    tracker = _ChunkProgress(len(results_dict), total_pages, progress_callback)

    if not missing_pages:
        all_parsing_results = restructure_parsing_results(
            [results_dict[p] for p in range(total_pages)],
            prettify_markdown=bool(options.get("prettify_markdown", True)),
            concatenate_pages=not bool(options.get("include_page_markers", True)),
        )
        return {
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {"layoutParsingResults": all_parsing_results},
        }

    if file_hash and fingerprint:
        write_partial_meta(file_hash, fingerprint, filename, total_pages, options)

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
                tracker,
                total_pages,
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
        concatenate_pages=not bool(options.get("include_page_markers", True)),
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
_MATH_HINT = re.compile(r"\\[a-zA-Z]+|[_^{}]|[=<>≤≥±∞∑∫√≠≈·×÷]|[+\-*/]")
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


def extract_markdown_from_response(
    api_response: dict,
    base_filename: str = "document",
    include_page_markers: bool = True,
) -> tuple[str, dict]:
    """
    Extract markdown text and images from API response.

    Rewrites image paths in the markdown to match the ZIP archive structure:
    - Original: imgs/img_xxx.jpg
    - Rewritten: {base_filename}_images/page_{n}/img_xxx.jpg (multi-page)
    - Rewritten: {base_filename}_images/img_xxx.jpg (single-page)

    When include_page_markers is True and there is more than one page, each
    page is prefixed with an HTML comment (`<!-- Page N -->`) for RAG splits.

    Args:
        api_response: The API response dictionary
        base_filename: Base filename for constructing image paths (without extension)
        include_page_markers: Insert <!-- Page N --> comments between pages

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

        if include_page_markers and len(parsing_results) > 1:
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


def _render_file_cache_banner(cached: dict) -> None:
    """Persistent per-file cache notice (re-rendered every script run)."""
    if cached.get("from_disk_cache"):
        st.success(
            "⚡ Instant from disk cache — previously processed with these options. "
            "Change processing options to reprocess."
        )
    else:
        st.info(
            "📋 Using cached results from this session. "
            "Change processing options to reprocess."
        )


def _flash_ocr_messages(messages: list[tuple[str, str]]) -> None:
    """Render worker/UI status lines with the matching Streamlit call."""
    for level, text in messages:
        if level == "info":
            st.info(text)
        elif level == "warning":
            st.warning(text)
        elif level == "error":
            st.error(text)
        elif level == "success":
            st.success(text)
        else:
            st.write(text)


def _render_ocr_job_progress(snapshot: dict) -> None:
    """Progress UI for a background job. Safe to call from a fragment."""
    _flash_ocr_messages(snapshot["messages"])
    if snapshot["status"] == "cancelling":
        st.warning(
            "⏹️ Cancellation requested — waiting for the current pages to finish."
        )
    name = snapshot["current_file"] or "document"
    current, total = snapshot["progress"]
    detail = (snapshot.get("detail") or "").strip()
    if total:
        stored = snapshot.get("bar_ratio")
        if stored is not None:
            ratio = min(1.0, max(0.0, float(stored)))
        else:
            ratio = min(1.0, max(0.0, current / total))
        if detail:
            if stored is not None and current < total:
                text = f"{name}: {current}/{total} saved · {detail}"
            else:
                text = f"{name}: {current}/{total} · {detail}"
        else:
            text = f"{name}: page {current}/{total}..."
        st.progress(ratio, text=text)
    else:
        st.caption(f"Working on {name}…")


def _ocr_job_poll_tick() -> None:
    """One progress refresh; used as a Streamlit fragment body."""
    live = st.session_state.get("ocr_job")
    if live is None:
        return
    snap = live.snapshot()
    _render_ocr_job_progress(snap)
    if snap["status"] not in ("running", "cancelling"):
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()


def _poll_ocr_job_until_idle() -> None:
    """Refresh progress without blocking the script on OCR (keeps Cancel live)."""
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        fragment(run_every=0.4)(_ocr_job_poll_tick)()
        return

    _ocr_job_poll_tick()
    live = st.session_state.get("ocr_job")
    if live is not None and live.snapshot()["status"] in ("running", "cancelling"):
        time.sleep(0.5)
        st.rerun()


def _job_is_active(job: OcrJob | None) -> bool:
    if job is None:
        return False
    return job.snapshot()["status"] in ("running", "cancelling")


def _sync_ocr_job_into_session() -> dict | None:
    """Copy worker results into session_state. Drop the job when it has finished."""
    job = st.session_state.get("ocr_job")
    if job is None:
        return None
    snap = job.snapshot()
    for file_key, result in snap["results"].items():
        st.session_state.processing_results[file_key] = result
    if snap["status"] in ("done", "cancelled", "error"):
        messages = list(snap["messages"])
        if snap["error"]:
            messages.append(("error", snap["error"]))
        if snap["status"] == "cancelled" and not any(
            "cancel" in text.lower() for _, text in messages
        ):
            messages.append(
                (
                    "warning",
                    "Processing cancelled. Completed pages were saved to disk. "
                    "Click Start OCR again with the same file and options to resume.",
                )
            )
        st.session_state.last_ocr_messages = messages
        st.session_state.is_processing = False
        st.session_state.ocr_job = None
        return None
    return snap


def _prepare_files_for_job(
    valid_files: list, options_fp: str
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Load session/disk cache on the script thread; return files that need the API."""
    to_process: list[dict] = []
    messages: list[tuple[str, str]] = []
    for uploaded_file, file_content, file_hash in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        cached = st.session_state.processing_results.get(file_key)
        if cached is not None and cached.get("fingerprint") == options_fp:
            if cached.get("from_disk_cache"):
                messages.append(
                    (
                        "info",
                        f"{uploaded_file.name}: restored from the disk cache "
                        "(change processing options to reprocess).",
                    )
                )
            else:
                messages.append(
                    (
                        "info",
                        f"{uploaded_file.name}: using cached results. Re-upload to reprocess.",
                    )
                )
            continue
        disk_hit = load_cached_result(file_hash, options_fp)
        if disk_hit is not None:
            st.session_state.processing_results[file_key] = disk_hit
            messages.append(
                (
                    "success",
                    f"{uploaded_file.name}: instant from disk cache — previously "
                    "processed with these options.",
                )
            )
            continue
        to_process.append(
            {
                "file_key": file_key,
                "name": uploaded_file.name,
                "content": file_content,
                "hash": file_hash,
                "is_pdf": uploaded_file.name.lower().endswith(".pdf"),
            }
        )
    return to_process, messages


def _process_one_file_for_job(
    job: OcrJob, spec: dict, options: dict, options_fp: str
) -> dict:
    """OCR one file on the worker thread. Must not call Streamlit APIs."""
    filename = spec["name"]
    file_content = spec["content"]
    file_hash = spec["hash"]
    is_pdf = spec["is_pdf"]
    start_time = time.time()
    page_count = 0

    if is_pdf:
        page_count = get_pdf_page_count(file_content)
        cached_pages = load_completed_pages(file_hash, options_fp)
        already_done = sum(1 for p in cached_pages if 0 <= p < page_count)
        if already_done and already_done < page_count:
            job.add_message(
                "info",
                f"{filename}: resuming {already_done}/{page_count} pages already "
                "cached. Only remaining pages will be sent to the API.",
            )
        elif already_done == page_count and page_count > 0:
            job.add_message(
                "info",
                f"{filename}: assembling {page_count} cached pages into the final result.",
            )
        job.set_progress(already_done, page_count)

        api_response = process_pdf_in_batches(
            file_content=file_content,
            filename=filename,
            progress_callback=job.set_progress,
            cancel_event=job.cancel_event,
            file_hash=file_hash,
            fingerprint=options_fp,
            **options,
        )
    else:
        if job.cancel_event.is_set():
            raise CancellationError("Processing cancelled")
        page_count = 1
        job.set_progress(0, 1)
        api_response = process_document(
            file_content=file_content,
            filename=filename,
            **_layout_api_options(options),
        )
        job.set_progress(1, 1)

    base_filename = display_stem(filename)
    markdown_text, images = extract_markdown_from_response(
        api_response,
        base_filename,
        include_page_markers=bool(options.get("include_page_markers", True)),
    )
    processing_time = time.time() - start_time
    job.add_message(
        "success", f"{filename}: processed in {processing_time:.1f} seconds"
    )

    if not save_result_to_disk(
        hash_str=file_hash,
        fingerprint=options_fp,
        display_name=filename,
        markdown_text=markdown_text,
        images=images,
        page_count=page_count,
        options=options,
    ):
        job.add_message(
            "warning",
            f"{filename}: result was not saved to the disk cache. "
            "Check that ./data is writable (set APP_UID/APP_GID in .env to "
            "your `id -u`/`id -g` and recreate the Streamlit container).",
        )
    return {
        "markdown": markdown_text,
        "images": images,
        "response": api_response,
        "display_name": filename,
        "fingerprint": options_fp,
    }


def _run_ocr_job(
    job: OcrJob, files: list[dict], options: dict, options_fp: str
) -> None:
    """Background OCR loop. Never calls Streamlit."""
    try:
        for spec in files:
            if job.cancel_event.is_set():
                job.add_message(
                    "warning", "Skipping remaining files due to cancellation."
                )
                break
            job.set_current_file(spec["name"])
            try:
                result = _process_one_file_for_job(job, spec, options, options_fp)
                job.store_result(spec["file_key"], result)
            except CancellationError as e:
                job.add_message("warning", str(e))
                if spec["is_pdf"]:
                    job.add_message(
                        "info",
                        "Completed pages were saved to disk. Click Start OCR again "
                        "with the same file and options to resume.",
                    )
                job.finish("cancelled")
                return
            except requests.Timeout:
                job.add_message(
                    "error",
                    f"{spec['name']}: timed out after {API_TIMEOUT} seconds. "
                    "Try processing smaller files or increase timeout.",
                )
                continue
            except requests.RequestException as e:
                job.add_message("error", f"{spec['name']}: network error: {e}")
                continue
            except RuntimeError as e:
                job.add_message("error", f"{spec['name']}: {e}")
                if spec["is_pdf"] and "no pages" not in str(e).lower():
                    job.add_message(
                        "info",
                        "Successfully processed pages were saved. Click Start OCR "
                        "again with the same file and options to resume.",
                    )
                continue
            except Exception as e:
                logger.exception("OCR failed for %s", spec["name"])
                job.add_message("error", f"{spec['name']}: unexpected error: {e}")
                continue

        if job.cancel_event.is_set():
            job.finish("cancelled")
        else:
            job.finish("done")
    except Exception as e:
        logger.exception("OCR job failed")
        job.finish("error", str(e))


def display_ocr_file_output(
    file_key: str,
    display_name: str,
    markdown_text: str,
    images: dict,
    options: dict,
    cached: dict,
) -> None:
    """Preview / raw markdown / download tabs for one finished file."""
    tab_preview, tab_raw, tab_download = st.tabs(
        ["📖 Preview", "📝 Raw Markdown", "💾 Download"]
    )

    with tab_preview:
        page_separators = markdown_text.count("\n\n---\n\n")
        total_pages = page_separators + 1 if page_separators > 0 else 1

        if total_pages > MAX_PREVIEW_PAGES:
            pages = markdown_text.split("\n\n---\n\n")
            truncated_md = "\n\n---\n\n".join(pages[:MAX_PREVIEW_PAGES])

            st.warning(
                f"⚠️ Showing preview of first {MAX_PREVIEW_PAGES} pages only "
                f"(document has {total_pages} pages). Use 'Raw Markdown' tab or download for full content."
            )
            st.markdown(truncated_md)

            with st.expander(f"📄 Show all {total_pages} pages (may be slow)"):
                st.markdown(markdown_text)
        else:
            st.markdown(markdown_text)

        if images:
            st.subheader("🖼️ Extracted Images")
            max_images_preview = MAX_PREVIEW_PAGES * 3
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
        base_filename = display_stem(display_name)
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=markdown_text.encode("utf-8"),
                file_name=f"{base_filename}.md",
                mime="text/markdown",
                key=f"dl_md_{file_key}",
            )

        with col2:
            if images:
                zip_data = create_download_zip(markdown_text, images, base_filename)
                st.download_button(
                    label="📦 Download ZIP (with images)",
                    data=zip_data,
                    file_name=f"{base_filename}_ocr_result.zip",
                    mime="application/zip",
                    key=f"dl_zip_{file_key}",
                )
            else:
                st.info("No images to include in ZIP")

    if options["visualize"]:
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
                    for idx, (img_name, img_data) in enumerate(output_images.items()):
                        if img_data:
                            with vis_cols[idx % 2]:
                                img_bytes = decode_base64_image(img_data)
                                st.image(
                                    img_bytes,
                                    caption=img_name.replace("_", " ").title(),
                                    width="stretch",
                                )


def display_processing_options() -> dict:
    """Display and collect processing options from sidebar."""
    st.sidebar.header("⚙️ Processing Options")

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
            value=USE_DOC_UNWARPING,
            key=f"opt_unwarp_{mode_key}_off",
            help="Correct curved or warped document images. Off by default in quality-first too — UVDoc unwarping can 500 the llama.cpp VLM.",
        ),
        "use_layout_detection": st.sidebar.checkbox(
            "Layout Detection",
            value=True if OCR_QUALITY_FIRST else USE_LAYOUT_DETECTION,
            key=f"opt_layout_{mode_key}",
            help="Detect document layout structure (recommended)",
        ),
        "use_chart_recognition": st.sidebar.checkbox(
            "Chart Recognition",
            value=USE_CHART_RECOGNITION,
            key=f"opt_chart_{mode_key}_off",
            help="Enable chart and diagram recognition. Off by default in quality-first too.",
        ),
        "prettify_markdown": st.sidebar.checkbox(
            "Prettify Markdown",
            value=PRETTIFY_MARKDOWN,
            help="Format markdown output for better readability",
        ),
        "include_page_markers": st.sidebar.checkbox(
            "Page comments",
            value=INCLUDE_PAGE_MARKERS,
            key=f"opt_pages_{mode_key}",
            help="Insert HTML comments such as <!-- Page 2 --> between pages. "
            "Needed for RAG chunking. In quality-first mode this also keeps "
            "pages separate instead of concatenating them into one document.",
        ),
        "visualize": st.sidebar.checkbox(
            "Show Visualization",
            value=VISUALIZE_RESULTS,
            help="Display intermediate processing results (slower)",
        ),
    }

    if OCR_QUALITY_FIRST:
        st.sidebar.info("Quality-first pipeline is on.")

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

    schedule_cache_cleanup()

    # Skip the health probe while a job is running so Cancel/progress reruns
    # are not blocked on a /health round-trip.
    if _job_is_active(st.session_state.get("ocr_job")):
        api_healthy = True
    else:
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

    if not _dir_is_writable(DATA_DIR):
        st.error(
            f"Result cache directory `{DATA_DIR}` is not writable "
            f"(container uid={os.getuid()} gid={os.getgid()}). "
            "Re-uploads will not hit the disk cache. Set `APP_UID`/`APP_GID` in "
            "`.env` to your host `id -u`/`id -g`, chown `./data` and `./logs` "
            "to that user, and recreate Streamlit: "
            "`docker compose up -d --build --force-recreate streamlit-app`."
        )

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
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "start_ocr_requested" not in st.session_state:
        st.session_state.start_ocr_requested = False
    if "ocr_job" not in st.session_state:
        st.session_state.ocr_job = None
    if "last_ocr_messages" not in st.session_state:
        st.session_state.last_ocr_messages = []

    _sync_ocr_job_into_session()
    if (
        st.session_state.is_processing
        and not st.session_state.start_ocr_requested
        and not _job_is_active(st.session_state.get("ocr_job"))
    ):
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

    # Hydrate session from disk on every rerun so a re-upload with the same
    # options shows cached results immediately (no extra Start OCR click).
    to_process, cache_messages = _prepare_files_for_job(valid_files, options_fp)

    # Preview section
    st.header("👁️ Document Preview")
    st.caption("Review your documents before processing")

    for uploaded_file, file_content, file_hash in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        cached = st.session_state.processing_results.get(file_key)
        cache_hit = (
            cached is not None and cached.get("fingerprint") == options_fp
        )
        title = f"📄 {uploaded_file.name}"
        if cache_hit:
            title += "  ·  ⚡ cached"

        with st.expander(title, expanded=True):
            if cache_hit:
                _render_file_cache_banner(cached)
            else:
                already_done = len(load_completed_pages(file_hash, options_fp))
                if already_done:
                    st.info(
                        f"♻️ {already_done} page(s) already checkpointed. "
                        "Click Start OCR to resume."
                    )
            display_file_preview(uploaded_file, file_content)

            # Store file content for later processing
            st.session_state.files_to_process[file_key] = {
                "name": uploaded_file.name,
                "content": file_content,
                "hash": file_hash,
                "fingerprint": options_fp,
            }

    # Start OCR / Cancel — on_click runs before widgets so Cancel is enabled
    # on the same rerun that starts the background job.
    st.header("🚀 Start Processing")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_clicked = st.button(
            "🔍 Start OCR",
            type="primary",
            disabled=st.session_state.is_processing,
            help="Process all uploaded documents with OCR",
            on_click=_on_start_ocr,
            key="start_ocr",
        )
    with col2:
        cancel_clicked = st.button(
            "⏹️ Cancel",
            type="secondary",
            disabled=not st.session_state.is_processing,
            help="Stop after the current API chunk finishes. Completed pages are kept.",
            on_click=_on_cancel_ocr,
            key="cancel_ocr",
        )
    with col3:
        job = st.session_state.get("ocr_job")
        if _job_is_active(job):
            snap = job.snapshot()
            if snap["status"] == "cancelling":
                st.caption("Cancelling — waiting for current pages…")
            else:
                name = snap["current_file"] or "document"
                current, total = snap["progress"]
                detail = (snap.get("detail") or "").strip()
                if total:
                    caption = f"Processing {name} ({current}/{total})"
                    if detail:
                        caption = f"{caption} — {detail}"
                    st.caption(caption)
                else:
                    st.caption(f"Processing {name}…")
        else:
            cached_count = len(valid_files) - len(to_process)
            if not to_process:
                st.caption(
                    f"All {len(valid_files)} document(s) are cached — "
                    "Start OCR is not needed"
                )
            elif cached_count:
                st.caption(
                    f"Ready to process {len(to_process)} document(s) "
                    f"({cached_count} cached)"
                )
            else:
                st.caption(f"Ready to process {len(valid_files)} document(s)")

    # on_click runs in a live app; the return value covers AppTest and
    # any runner that skips callbacks.
    if start_clicked:
        st.session_state.is_processing = True
        st.session_state.start_ocr_requested = True
    if cancel_clicked:
        _on_cancel_ocr()

    if st.session_state.start_ocr_requested:
        st.session_state.start_ocr_requested = False
        st.session_state.last_ocr_messages = []
        if not to_process:
            st.session_state.is_processing = False
            st.session_state.last_ocr_messages = cache_messages
        else:
            job = OcrJob()
            for level, text in cache_messages:
                job.add_message(level, text)
            st.session_state.ocr_job = job
            threading.Thread(
                target=_run_ocr_job,
                args=(job, to_process, dict(options), options_fp),
                daemon=True,
                name="ocr-job",
            ).start()

    if _job_is_active(st.session_state.get("ocr_job")):
        st.header("🔄 Processing Results")
        _poll_ocr_job_until_idle()
        st.markdown("---")
        st.markdown(
            "Built with [Streamlit](https://streamlit.io) and "
            "[PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR)"
        )
        return

    # Results persist across reruns (tabs, downloads) instead of only the Start click.
    st.header("🔄 Processing Results")
    if st.session_state.last_ocr_messages:
        _flash_ocr_messages(st.session_state.last_ocr_messages)
        st.session_state.last_ocr_messages = []

    result_items: list[tuple[str, dict]] = []
    for uploaded_file, _, _ in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        cached = st.session_state.processing_results.get(file_key)
        if cached is None or cached.get("fingerprint") != options_fp:
            continue
        result_items.append((file_key, cached))
        markdown_text = normalize_markdown_math(cached["markdown"])
        cached["markdown"] = markdown_text
        result_title = f"📄 {uploaded_file.name}"
        if cached.get("from_disk_cache"):
            result_title += "  ·  ⚡ cached"
        with st.expander(result_title, expanded=True):
            if cached.get("from_disk_cache"):
                _render_file_cache_banner(cached)
            display_ocr_file_output(
                file_key,
                cached.get("display_name", uploaded_file.name),
                markdown_text,
                cached["images"],
                options,
                cached,
            )

    if not result_items:
        st.info("👆 Click 'Start OCR' to begin processing your documents")

    if len(result_items) > 1:
        st.header("📦 Batch Download")
        if st.button("Download All Results as ZIP", key="batch_zip_prepare"):
            batch_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                batch_zip_buffer, "w", zipfile.ZIP_DEFLATED
            ) as batch_zip:
                for file_key, data in result_items:
                    base_name = display_stem(
                        data.get("display_name", file_key.rsplit("_", 1)[0])
                    )
                    batch_zip.writestr(
                        f"{base_name}/{base_name}.md", data["markdown"].encode("utf-8")
                    )
                    images_dir = f"{base_name}_images"
                    for img_path, img_data in data["images"].items():
                        img_bytes = decode_base64_image(img_data)
                        batch_zip.writestr(
                            f"{base_name}/{images_dir}/{img_path}", img_bytes
                        )

            batch_zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Results",
                data=batch_zip_buffer.getvalue(),
                file_name="all_ocr_results.zip",
                mime="application/zip",
                key="batch_zip_download",
            )

    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io) and "
        "[PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR)"
    )


if __name__ == "__main__":
    main()
