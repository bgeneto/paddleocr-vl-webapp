"""
PaddleOCR-VL Document Parser Streamlit Application

This application provides a web interface for document OCR using PaddleOCR-VL
with vLLM backend for production-ready inference.
"""

import base64
import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
# Parallel workers - keep low since API may serialize requests anyway
MAX_PARALLEL_PAGES = int(os.getenv("MAX_PARALLEL_PAGES", "2"))
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

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]


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


def process_chunk(args: tuple) -> tuple[int, int, list | None, str | None]:
    """
    Process a chunk of PDF pages. Helper function for parallel chunk processing.

    Each chunk contains multiple pages, which allows vLLM to batch-process them
    together for better GPU utilization.

    Args:
        args: Tuple of (chunk_index, start_page, end_page, chunk_content, filename, options, cancel_event)

    Returns:
        Tuple of (chunk_index, start_page, parsing_results_list, error_message)
    """
    chunk_index, start_page, end_page, chunk_content, filename, options, cancel_event = args
    num_pages_in_chunk = end_page - start_page + 1

    # Check for cancellation before starting
    if cancel_event and cancel_event.is_set():
        return (chunk_index, start_page, None, "Cancelled")

    try:
        # Send multi-page chunk to API - vLLM will batch-process all pages together
        chunk_response = process_document(
            file_content=chunk_content,
            filename=filename,
            **options,
        )
        result = chunk_response.get("result", {})
        parsing_results = result.get("layoutParsingResults", [])

        # Each parsing result corresponds to a page in the chunk
        return (chunk_index, start_page, parsing_results, None)
    except Exception as e:
        return (chunk_index, start_page, None, str(e))


def process_pdf_in_batches(
    file_content: bytes,
    filename: str,
    progress_callback=None,
    cancel_event: threading.Event = None,
    max_workers: int = None,
    pages_per_chunk: int = None,
    **options,
) -> dict:
    """
    Process a PDF document with chunked parallel processing for optimal GPU utilization.

    Instead of processing single pages, this splits the PDF into multi-page chunks
    (e.g., 8 pages each). Each chunk is sent to the API as a multi-page PDF, allowing
    vLLM to batch-process all pages in the chunk together. This dramatically improves
    GPU utilization compared to single-page processing.

    Args:
        file_content: Raw bytes of the PDF file
        filename: Original filename
        progress_callback: Optional callback function(completed_pages, total_pages)
        cancel_event: Optional threading.Event to signal cancellation
        max_workers: Maximum parallel chunk workers (default: MAX_PARALLEL_PAGES)
        pages_per_chunk: Pages per chunk for GPU batching (default: PAGES_PER_CHUNK)
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

    # For small PDFs, adjust chunk size to avoid overhead
    # e.g., 12 pages with chunk=8 → better to do 6+6 than 8+4
    num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
    if num_chunks > 1:
        # Distribute pages more evenly across chunks
        adjusted_chunk_size = (total_pages + num_chunks - 1) // num_chunks
        pages_per_chunk = max(1, adjusted_chunk_size)

    # Split PDF into multi-page chunks
    chunks = split_pdf_into_chunks(file_content, pages_per_chunk)
    num_chunks = len(chunks)

    # Prepare arguments for parallel chunk processing
    process_args = [
        (chunk_idx, start_page, end_page, chunk_bytes, filename, options, cancel_event)
        for chunk_idx, (start_page, end_page, chunk_bytes) in enumerate(chunks)
    ]

    # Process chunks in parallel (each chunk is batch-processed by vLLM)
    results_dict = {}  # page_num -> parsing_result
    errors = []
    completed_pages = 0
    cancelled = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk tasks
        future_to_chunk = {
            executor.submit(process_chunk, args): (args[0], args[1], args[2])  # chunk_idx, start, end
            for args in process_args
        }

        # Collect results as chunks complete
        for future in as_completed(future_to_chunk):
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                cancelled = True
                for f in future_to_chunk:
                    f.cancel()
                break

            chunk_idx, start_page, parsing_results, error = future.result()
            _, chunk_start, chunk_end = future_to_chunk[future]
            pages_in_chunk = chunk_end - chunk_start + 1
            completed_pages += pages_in_chunk

            if progress_callback:
                progress_callback(completed_pages, total_pages)

            if error:
                if error == "Cancelled":
                    cancelled = True
                else:
                    errors.append(f"Chunk {chunk_idx + 1} (pages {chunk_start + 1}-{chunk_end + 1}): {error}")
            elif parsing_results:
                # Map each parsing result to its actual page number
                for i, pr in enumerate(parsing_results):
                    actual_page_num = start_page + i
                    results_dict[actual_page_num] = pr

    # Handle cancellation
    if cancelled:
        raise CancellationError(
            f"Processing cancelled after {completed_pages} of {total_pages} pages"
        )

    # Report any errors
    if errors:
        error_summary = "; ".join(errors[:5])
        if len(errors) > 5:
            error_summary += f" ... and {len(errors) - 5} more errors"
        raise RuntimeError(f"Some chunks failed to process: {error_summary}")

    # Combine results in page order
    all_parsing_results = []
    for page_num in sorted(results_dict.keys()):
        all_parsing_results.append(results_dict[page_num])

    # Combine all results into a single response structure
    combined_response = {
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": all_parsing_results,
        },
    }

    return combined_response


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
    return full_markdown, all_images


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

    options = {
        "use_doc_orientation_classify": st.sidebar.checkbox(
            "Document Orientation Classification",
            value=USE_DOC_ORIENTATION_CLASSIFY,
            help="Automatically detect and correct document orientation",
        ),
        "use_doc_unwarping": st.sidebar.checkbox(
            "Document Unwarping",
            value=USE_DOC_UNWARPING,
            help="Correct curved or warped document images",
        ),
        "use_layout_detection": st.sidebar.checkbox(
            "Layout Detection",
            value=USE_LAYOUT_DETECTION,
            help="Detect document layout structure (recommended)",
        ),
        "use_chart_recognition": st.sidebar.checkbox(
            "Chart Recognition",
            value=USE_CHART_RECOGNITION,
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
        valid_files.append((uploaded_file, file_content))

    if not valid_files:
        return

    # Preview section
    st.header("👁️ Document Preview")
    st.caption("Review your documents before processing")

    for uploaded_file, file_content in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            display_file_preview(uploaded_file, file_content)

            # Store file content for later processing
            st.session_state.files_to_process[file_key] = {
                "name": uploaded_file.name,
                "content": file_content,
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

    for uploaded_file, file_content in valid_files:
        # Check for cancellation between files
        if st.session_state.cancel_requested:
            cancel_event.set()
            st.warning("⏹️ Skipping remaining files due to cancellation.")
            break

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        is_pdf = uploaded_file.name.lower().endswith(".pdf")

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            # Check if already processed
            if file_key in st.session_state.processing_results:
                cached = st.session_state.processing_results[file_key]
                st.info("📋 Using cached results. Re-upload to reprocess.")
                markdown_text = cached["markdown"]
                images = cached["images"]
            else:
                try:
                    start_time = time.time()

                    if is_pdf:
                        # Use batch processing for PDFs to handle all pages
                        page_count = get_pdf_page_count(file_content)

                        # Create containers for progress and cancel status
                        progress_container = st.empty()
                        progress_bar = progress_container.progress(
                            0,
                            text=f"Processing page 0/{page_count}... (Cancel with button above)",
                        )

                        def update_progress(current, total):
                            # Check cancel state from session
                            if st.session_state.cancel_requested:
                                cancel_event.set()
                            progress_bar.progress(
                                current / total,
                                text=f"Processing page {current}/{total}...",
                            )

                        api_response = process_pdf_in_batches(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            progress_callback=update_progress,
                            cancel_event=cancel_event,
                            **options,
                        )
                        progress_container.empty()
                    else:
                        # Process single image directly
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            api_response = process_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                **options,
                            )

                    # Extract markdown with image paths rewritten to match ZIP structure
                    base_filename = Path(uploaded_file.name).stem
                    markdown_text, images = extract_markdown_from_response(api_response, base_filename)

                    processing_time = time.time() - start_time
                    st.success(f"✅ Processed in {processing_time:.1f} seconds")

                    # Cache results
                    st.session_state.processing_results[file_key] = {
                        "markdown": markdown_text,
                        "images": images,
                        "response": api_response,
                    }

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
                    st.session_state.is_processing = False
                    st.stop()  # Stop further processing
                except RuntimeError as e:
                    st.error(f"❌ Processing error: {str(e)}")
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
                base_filename = Path(uploaded_file.name).stem

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
                    base_name = file_key.rsplit("_", 1)[0]  # Remove size suffix
                    base_name = Path(base_name).stem
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
